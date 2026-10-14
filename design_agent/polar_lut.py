"""
Polar look-up tables (LUTs): a NeuralFoil compatibility layer.

Why this exists
---------------
NeuralFoil's *CasADi* evaluation path (NF 0.3.3 + CasADi 3.8.1, this repo's
pinned stack) returns values that disagree with its numeric path by far more
than interpolation error (e.g. CL 0.48 vs 1.14 for a NACA 2412 at alpha=2 deg),
so tracing `Airfoil.get_aero_from_neuralfoil` inside an `asb.Opti` problem is
not trustworthy. The *numeric* path is fine.

Strategy
--------
Numerically evaluate NeuralFoil once on a structured (alpha, Re, mach) grid for
the current airfoil, build CasADi `interpolant` linear LUTs for CL/CD/CM, and —
only while a symbolic problem is being assembled — monkey-patch
`Airfoil.get_aero_from_neuralfoil` to answer from the LUT. Numeric calls always
fall through to the real NeuralFoil, so verification (post-solve, Monte-Carlo)
uses the true model. The patch is context-managed and always restored; the
`aerosandbox` package itself is never modified on disk.

For a frozen (database) airfoil the LUT is built once. For CST-shape
optimization the optimizer rebuilds the LUT at each outer fixed-point iteration,
so the design converges to a point where LUT and NeuralFoil agree.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import casadi as cas

import aerosandbox as asb

# Finer alpha spacing through the CLmax region; mach nodes dense around the
# typical Mcrit cliff where NF transitions sharply (see PolarLUT docstring).
ALPHA_GRID = np.concatenate(
    [np.linspace(-10.0, 6.0, 17), np.linspace(6.5, 20.0, 28), [22.5, 25.0]]
)
RE_GRID = np.logspace(4.7, 7.7, 15)  # ~5e4 .. 5e7
MACH_GRID = np.array([0.0, 0.1, 0.175, 0.25, 0.325, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])


def _flatten_fortran(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values.ravel(order="F"))


class PolarLUT:
    """CL/CD/CM as linear CasADi interpolants over (alpha, Re, mach)."""

    def __init__(self, airfoil: asb.Airfoil, model_size: str = "large") -> None:
        self.airfoil_name = getattr(airfoil, "name", "airfoil")
        # Full-grid vectorized NeuralFoil evaluation (numeric only).
        ma, re, al = np.meshgrid(MACH_GRID, RE_GRID, ALPHA_GRID, indexing="ij")
        aero = airfoil.get_aero_from_neuralfoil(
            alpha=al.ravel(), Re=re.ravel(), mach=ma.ravel(),
            model_size=model_size,
        )
        shape = (len(MACH_GRID), len(RE_GRID), len(ALPHA_GRID))
        self._interp: dict[str, Any] = {}
        for key in ("CL", "CD", "CM"):
            vals = np.asarray(aero[key], dtype=float).reshape(shape)
            self._interp[key] = cas.interpolant(
                f"nf_lut_{key}", "linear",
                [MACH_GRID.tolist(), RE_GRID.tolist(), ALPHA_GRID.tolist()],
                _flatten_fortran(vals),
            )

    def evaluate(self, alpha: Any, re: Any, mach: Any = 0.0) -> dict[str, Any]:
        """Elementwise LUT evaluation preserving the broadcast input shape."""
        def shape_of(value: Any) -> tuple[int, ...]:
            if isinstance(value, (cas.MX, cas.SX, cas.DM)):
                return tuple(int(d) for d in value.shape)
            return tuple(np.asarray(value).shape)

        target_shape = np.broadcast_shapes(
            shape_of(alpha), shape_of(re), shape_of(mach)
        )

        a = _vec(alpha)
        r = _vec(re)
        m = _vec(mach)
        n = max(a.shape[0], r.shape[0], m.shape[0])
        a = _bc(a, n)
        r = _bc(r, n)
        m = _bc(m, n)
        symbolic = any(isinstance(v, (cas.MX, cas.SX)) for v in (alpha, re, mach))
        out: dict[str, Any] = {}

        for key in ("CL", "CD", "CM"):
            interp = self._interp[key]
            col = [interp(cas.vertcat(m[i], r[i], a[i])) for i in range(n)]
            flat = cas.vertcat(*col)

            if symbolic:
                if not target_shape:
                    out[key] = flat[0]
                else:
                    rows = target_shape[0]
                    cols = (
                        int(np.prod(target_shape[1:]))
                        if len(target_shape) > 1
                        else 1
                    )
                    out[key] = cas.reshape(flat, rows, cols)
            else:
                arr = np.asarray(flat)
                out[key] = arr.reshape(target_shape) if target_shape else arr.ravel()

        return out

    @property
    def coverage(self) -> dict[str, tuple[float, float]]:
        return {
            "alpha": (float(ALPHA_GRID.min()), float(ALPHA_GRID.max())),
            "Re": (float(RE_GRID.min()), float(RE_GRID.max())),
            "mach": (float(MACH_GRID.min()), float(MACH_GRID.max())),
        }


_LUT_CACHE: dict[tuple[str, str], "PolarLUT"] = {}


def build_lut_registry(
    airfoils: list, model_size: str = "large"
) -> "dict[str, PolarLUT]":
    """One LUT per airfoil, keyed by airfoil name.

    A full airplane references more than one airfoil (design wing airfoil plus
    the naca0010 empennage), and routing every section through a single
    design-airfoil LUT invents phantom tail lift. Names are sufficient keys
    here because the airplane is always assembled by
    `DesignVector.to_airplane`; a lookup miss raises `KeyError` loudly rather
    than silently using wrong polars.
    """
    registry: dict[str, PolarLUT] = {}
    for af in airfoils:
        key = getattr(af, "name", None) or "airfoil"
        if key not in registry:
            registry[key] = PolarLUT(af, model_size=model_size)
    return registry


def build_design_registry(design: Any, model_size: str = "large") -> "dict[str, PolarLUT]":
    """LUT registry for a `DesignVector`: its wing airfoil + the naca0010 tails.

    For a frozen (database) airfoil the LUTs are cached: the grid never changes
    across solves/repairs, and rebuilding costs tens of seconds. CST-shape
    optimization (``frozen_airfoil is None``) always rebuilds, because the
    airfoil itself is a design variable there.
    """
    if getattr(design, "frozen_airfoil", None) is None:
        return build_lut_registry(
            [design.airfoil(), asb.Airfoil("naca0010")], model_size=model_size
        )
    registry: dict[str, PolarLUT] = {}
    for af in [design.airfoil(), asb.Airfoil("naca0010")]:
        key = getattr(af, "name", None) or "airfoil"
        cache_key = (key, model_size)
        if cache_key not in _LUT_CACHE:
            _LUT_CACHE[cache_key] = PolarLUT(af, model_size=model_size)
        registry[key] = _LUT_CACHE[cache_key]
    return registry


def _resolve_lut(lut: Any, airfoil: Any) -> "PolarLUT | None":
    """Single LUT (legacy), mapping by airfoil name, or callable -> LUT/None."""
    if isinstance(lut, PolarLUT):
        return lut
    if callable(lut):
        return lut(airfoil)
    key = getattr(airfoil, "name", None) or "airfoil"
    return lut[key]


def _is_casadi(value: Any) -> bool:
    return isinstance(value, (cas.MX, cas.SX, cas.DM))


def _vec(value: Any) -> Any:
    """Column-vector (casadi) or 1-d float array."""
    if isinstance(value, (cas.MX, cas.SX, cas.DM)):
        v = cas.vec(value)
        return v
    return np.atleast_1d(np.asarray(value, dtype=float).ravel())


def _bc(v: Any, n: int) -> Any:
    """Broadcast a length-1 column to length n."""
    if v.shape[0] == n:
        return v
    if isinstance(v, (cas.MX, cas.SX, cas.DM)):
        return cas.repmat(v, n) if v.shape[0] == 1 else v
    return np.repeat(v, n) if v.shape[0] == 1 else v


@contextmanager
def patched_airfoil_aero(lut: "PolarLUT | dict[str, PolarLUT] | Any"):
    """
    Temporarily route `Airfoil.get_aero_from_neuralfoil` through LUT(s).

    `lut` may be a single `PolarLUT` (all intercepted airfoils), a
    name->LUT mapping/registry (per-airfoil dispatch, see
    `build_lut_registry`), or a callable `Airfoil -> PolarLUT | None`.

    Only calls whose alpha/Re/mach contain CasADi symbols are intercepted;
    numeric calls use the genuine NeuralFoil. `analysis_confidence` is NOT
    provided by the LUT (symbolic mode): callers should treat it as a
    numeric-verification quantity.
    """
    original = asb.Airfoil.get_aero_from_neuralfoil

    def patched(self, alpha, Re, mach=0.0, n_crit=9.0, xtr_upper=1.0,
                xtr_lower=1.0, model_size="large", control_surfaces=None,
                include_360_deg_effects=True):
        if _is_casadi(alpha) or _is_casadi(Re) or _is_casadi(mach):
            resolved = _resolve_lut(lut, self)
            if resolved is None:  # unknown airfoil: genuine (broken-casadi) NF
                return original(
                    self, alpha=alpha, Re=Re, mach=mach, n_crit=n_crit,
                    xtr_upper=xtr_upper, xtr_lower=xtr_lower,
                    model_size=model_size, control_surfaces=control_surfaces,
                    include_360_deg_effects=include_360_deg_effects,
                )
            out = resolved.evaluate(alpha=alpha, re=Re, mach=mach)
            # mimic the full NeuralFoil dict shape with NaN confidence
            out["analysis_confidence"] = float("nan")
            return out
        return original(
            self, alpha=alpha, Re=Re, mach=mach, n_crit=n_crit,
            xtr_upper=xtr_upper, xtr_lower=xtr_lower, model_size=model_size,
            control_surfaces=control_surfaces,
            include_360_deg_effects=include_360_deg_effects,
        )

    asb.Airfoil.get_aero_from_neuralfoil = patched
    try:
        yield
    finally:
        asb.Airfoil.get_aero_from_neuralfoil = original
