"""
Monte-Carlo robustness verification — the uncertainty-aware layer.

The optimizer produces a *nominal* design; this module answers the question the
research actually cares about: does the design still satisfy the constraint
catalog when reality deviates from the model? Each Monte-Carlo sample disperses

  * geometry     — manufacturing tolerance, via ``DesignVector.perturb``
  * CG           — loading uncertainty, additive jitter on ``cg_mac_fraction``
  * empty mass   — weight growth, lognormal ``mass_factor`` (see ``Dispersion``)
  * air density  — first-order force scaling (hot/high-day surrogate)
  * temperature  — exact ISA deviation via ``asb.Atmosphere``

Every sample is re-trimmed (Newton on the cruise/loiter angles of attack
against the dispersed weight — an untrimmed sample would fail trim trivially
and the statistics would be meaningless) and verified with the *true numeric*
physics stack: numeric NeuralFoil, full AeroBuildup, dynamic modes — the same
ground truth the optimizer's post-solve verification uses.

Typical use
-----------
>>> report = run_monte_carlo(result.design, spec,
...                          alphas=result.solver_stats.get("alphas"),
...                          n_samples=64, seed=0)
>>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as onp

from design_agent.analysis import Dispersion, G, analyze, scalar
from design_agent.constraints import (
    ConstraintRecord,
    Margins,
    all_passed,
    evaluate_numeric,
)
from design_agent.mission_spec import MissionSpec
from design_agent.parameterization import DesignVector

CG_BOUNDS = (0.05, 0.45)  # keep dispersed CG inside the optimizer's box
ALPHA_BOUNDS = (-5.0, 16.0)  # match the optimizer's trim-variable box [deg]


@dataclass(frozen=True)
class UncertaintyModel:
    """Sampling widths for the Monte-Carlo dispersion (1-sigma values).

    geometry_sigma       relative, on dimensional geometry (manufacturing)
    cg_sigma_mac         additive, in MAC fractions (loading/CG uncertainty)
    mass_factor_sigma    lognormal sigma on the empty-mass model (weight growth)
    density_sigma        relative, first-order aero-force scaling
    temperature_sigma_k  ISA deviation [K]
    """

    geometry_sigma: float = 0.01
    cg_sigma_mac: float = 0.01
    mass_factor_sigma: float = 0.05
    density_sigma: float = 0.02
    temperature_sigma_k: float = 3.0


@dataclass
class SampleResult:
    """One Monte-Carlo sample, fully verified against the constraint catalog."""

    index: int  # 0 = nominal (undispersed) when include_nominal=True
    design: DesignVector
    dispersion: Dispersion
    mass_total_kg: float
    records: list[ConstraintRecord]
    success: bool
    alphas: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class RobustnessReport:
    """Aggregate Monte-Carlo statistics for one nominal design."""

    samples: list[SampleResult]
    model: UncertaintyModel

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_success(self) -> int:
        return sum(1 for s in self.samples if s.success)

    @property
    def p_success(self) -> float:
        """Probability that a dispersed aircraft still passes everything."""
        if not self.samples:
            return 0.0
        return self.n_success / self.n_samples

    @property
    def nominal(self) -> SampleResult | None:
        return self.samples[0] if self.samples else None

    def pass_probability(self) -> dict[str, float]:
        """Per-constraint pass probability over the samples that produced it."""
        votes: dict[str, list[bool]] = {}
        for s in self.samples:
            for r in s.records:
                votes.setdefault(r.name, []).append(r.passed)
        return {k: sum(v) / len(v) for k, v in votes.items()}

    def mass_percentiles(self) -> dict[str, float]:
        m = onp.array(
            [s.mass_total_kg for s in self.samples], dtype=float
        )
        m = m[onp.isfinite(m)]
        if m.size == 0:
            return {"p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
        return {
            "p5": float(onp.percentile(m, 5)),
            "p50": float(onp.percentile(m, 50)),
            "p95": float(onp.percentile(m, 95)),
        }

    def binding_constraints(self) -> list[tuple[str, float]]:
        """(name, pass_probability) sorted weakest first."""
        return sorted(self.pass_probability().items(), key=lambda kv: kv[1])

    def failed_sample_messages(self) -> list[str]:
        return [f"sample {s.index}: {s.message}" for s in self.samples
                if s.message and not s.records]

    def summary(self) -> str:
        lines = [
            f"Monte-Carlo robustness: {self.n_samples} samples "
            f"({self.n_success} fully feasible)",
            f"  p(all constraints satisfied) = {self.p_success:.2f}",
        ]
        mp = self.mass_percentiles()
        lines.append(
            f"  gross mass p5/p50/p95 = {mp['p5']:.0f} / {mp['p50']:.0f} / "
            f"{mp['p95']:.0f} kg"
        )
        lines.append("  constraint pass probabilities (weakest first):")
        for name, p in self.binding_constraints():
            lines.append(f"    {name:22s} {p:.2f}")
        for msg in self.failed_sample_messages():
            lines.append(f"  analysis failure — {msg}")
        return "\n".join(lines)


def run_monte_carlo(
    design: DesignVector,
    spec: MissionSpec,
    margins: Margins | None = None,
    model: UncertaintyModel | None = None,
    n_samples: int = 32,
    seed: int = 0,
    alphas: dict[str, Any] | None = None,
    trim_iterations: int = 2,
    compute_modes: bool = True,
    include_nominal: bool = True,
) -> RobustnessReport:
    """Disperse ``design`` and verify each sample against the constraint catalog.

    Parameters
    ----------
    design
        Nominal design point (e.g. ``OptimizationResult.design``).
    spec, margins
        Mission and constraint catalog; margins default like the optimizer's.
    model
        Sampling widths; defaults are mild, physically-motivated dispersions.
    n_samples
        Number of *dispersed* samples (the nominal point is extra).
    seed
        RNG seed — results are exactly reproducible for a fixed seed.
    alphas
        Baseline trim angles [deg], e.g. ``result.solver_stats["alphas"]``.
        Each sample is re-trimmed from this starting point.
    trim_iterations
        Newton iterations per sample for the cruise/loiter trim.
    compute_modes
        Include dynamic-mode constraints in each sample's verification.
    include_nominal
        Evaluate the undispersed design as sample 0.
    """
    model = model or UncertaintyModel()
    margins = (margins or Margins()).with_overrides(spec)
    base_alphas = {
        k: float(scalar(v))
        for k, v in (alphas or {"cruise": 2.0, "loiter": 2.0, "sea": 8.0}).items()
    }
    rng = onp.random.default_rng(seed)

    samples: list[SampleResult] = []
    if include_nominal:
        samples.append(
            _eval_sample(0, design, Dispersion(), spec, margins,
                         base_alphas, trim_iterations, compute_modes)
        )
    n_total = n_samples + len(samples)
    for k in range(len(samples), n_total):
        d = design.perturb(rng, geometry_sigma=model.geometry_sigma)
        cg = float(onp.clip(
            d.cg_mac_fraction + rng.normal(0.0, model.cg_sigma_mac),
            *CG_BOUNDS,
        ))
        d = replace(d, cg_mac_fraction=cg)
        disp = Dispersion(
            mass_factor=float(onp.exp(rng.normal(0.0, model.mass_factor_sigma))),
            density_factor=float(max(0.7, rng.normal(1.0, model.density_sigma))),
            temperature_offset_k=float(rng.normal(0.0, model.temperature_sigma_k)),
        )
        samples.append(
            _eval_sample(k, d, disp, spec, margins,
                         base_alphas, trim_iterations, compute_modes)
        )
    return RobustnessReport(samples=samples, model=model)


def _eval_sample(
    index: int,
    design: DesignVector,
    dispersion: Dispersion,
    spec: MissionSpec,
    margins: Margins,
    base_alphas: dict[str, float],
    trim_iterations: int,
    compute_modes: bool,
) -> SampleResult:
    """Re-trim and verify one (possibly dispersed) design point."""
    try:
        res, alphas = _retrim(
            design, spec, dispersion, base_alphas, trim_iterations, compute_modes
        )
        records = evaluate_numeric(res, margins)
        return SampleResult(
            index=index,
            design=design,
            dispersion=dispersion,
            mass_total_kg=scalar(res.mass.total),
            records=records,
            success=all_passed(records),
            alphas=alphas,
        )
    except Exception as exc:
        return SampleResult(
            index=index,
            design=design,
            dispersion=dispersion,
            mass_total_kg=float("nan"),
            records=[],
            success=False,
            message=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _retrim(
    design: DesignVector,
    spec: MissionSpec,
    dispersion: Dispersion,
    base_alphas: dict[str, float],
    trim_iterations: int,
    compute_modes: bool,
):
    """Newton-trim cruise & loiter alpha so L = W at the dispersed point.

    Uses the 3-D lift-curve slope from the stability-derivative run:
    dL/d(alpha_deg) = (L/CL) * CLa * pi/180. One or two iterations suffice for
    the small dispersions used here. Returns (AnalysisResult, alphas).
    """
    a_c = float(base_alphas.get("cruise", 2.0))
    a_l = float(base_alphas.get("loiter", a_c))
    a_sea = float(base_alphas.get("sea", 8.0))

    res = analyze(
        design, spec,
        alphas={"cruise": a_c, "loiter": a_l, "sea": a_sea},
        dispersion=dispersion,
        compute_modes=compute_modes,
    )
    for _ in range(trim_iterations):
        w = scalar(res.mass.total) * G
        try:
            cla = scalar(res.aero["CLa"][0])  # per radian
        except (KeyError, IndexError, TypeError):
            break
        if not onp.isfinite(cla) or abs(cla) < 1e-6:
            break

        l_c = scalar(res.aero["L"][0])
        if abs(l_c - w) <= 1e-3 * w:
            break  # converged
        deltas = {}
        for idx, name in ((0, "cruise"), (1, "loiter")):
            L = scalar(res.aero["L"][idx])
            cl = scalar(res.aero["CL"][idx])
            if not (onp.isfinite(L) and onp.isfinite(cl)) or abs(cl) < 1e-6:
                continue
            dL_da = (L / cl) * cla * onp.pi / 180.0  # N per degree
            deltas[name] = float(onp.clip((w - L) / dL_da, -5.0, 5.0))
        a_c = float(onp.clip(a_c + deltas.get("cruise", 0.0), *ALPHA_BOUNDS))
        a_l = float(onp.clip(a_l + deltas.get("loiter", 0.0), *ALPHA_BOUNDS))
        res = analyze(
            design, spec,
            alphas={"cruise": a_c, "loiter": a_l, "sea": a_sea},
            dispersion=dispersion,
            compute_modes=compute_modes,
        )
    return res, {"cruise": a_c, "loiter": a_l, "sea": a_sea}
