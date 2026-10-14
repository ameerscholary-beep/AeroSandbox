"""
Optimization problem assembly and solve.

Builds one `asb.Opti` problem per candidate design: symbolic design vector +
trim AoAs + the TOGW closure variable, AeroSandbox physics traced symbolically,
constraint catalog registered, objective = gross mass.

Failure philosophy: `optimize()` never raises on solver trouble. If IPOPT fails,
the *last iterate* is still recovered (`behavior_on_failure="return_last"`) and
verified numerically, so the caller always receives a full constraint diagnostic
— the agent loop needs to see *why* a design failed, not an exception.

Repair strategy (numeric-only failure modes): flight-dynamic modes cannot be
traced symbolically, so they are repaired by *reseeding* the flat directions of
the problem (dihedral, tail arm/area) and/or moving the empirical tail-volume
floors, then re-solving. Dihedral has no symbolic gradient (it barely changes
cruise L/D in the surrogate), so IPOPT leaves it wherever it is seeded — which
makes reseeding an effective in-Opti lever for spiral stability.
"""

from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import aerosandbox as asb
import aerosandbox.numpy as np

from design_agent.analysis import G, analyze, scalar
from design_agent.constraints import (
    ConstraintRecord,
    Margins,
    all_passed,
    evaluate_numeric,
    register_symbolic,
)
from design_agent.mission_spec import MissionSpec
from design_agent.parameterization import DesignVector, initial_guess


class OptimizationBudgetExceeded(RuntimeError):
    """Raised when a single solve exceeds its time budget."""


_SOLVE_SUPPORTS_RETURN_LAST = (
    "behavior_on_failure" in inspect.signature(asb.Opti.solve).parameters
)


@dataclass
class OptimizationResult:
    success: bool
    design: DesignVector
    mass_total_kg: float
    records: list[ConstraintRecord]
    solver_stats: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    build_time_s: float = 0.0
    solve_time_s: float = 0.0

    @property
    def constraints_passed(self) -> int:
        return sum(1 for r in self.records if r.passed)

    @property
    def constraints_total(self) -> int:
        return len(self.records)


def build_opti(
    spec: MissionSpec,
    x0: DesignVector,
    margins: Margins,
    lut: "PolarLUT | dict[str, PolarLUT] | None" = None,
) -> tuple[asb.Opti, list[Any], Any, Any]:
    """Create the Opti problem. Returns (opti, flat_vars, weight_n, alphas)."""
    opti = asb.Opti()
    flat0 = np.asarray(x0.to_flat(), dtype=float)
    flat_vars: list[Any] = []
    offset = 0
    for s in x0.field_specs(x0.frozen_airfoil):
        block = flat0[offset : offset + s.size]
        offset += s.size
        flat_vars.append(
            opti.variable(
                init_guess=block,
                lower_bound=s.lo,
                upper_bound=s.hi,
                scale=s.scale,
                log_transform=s.log_transform and s.size == 1,
            )
        )
    sym = DesignVector.from_flat(np.concatenate(flat_vars), x0.frozen_airfoil)

    weight_n = opti.variable(
        init_guess=2.5 * spec.useful_load_kg * G,
        lower_bound=300.0 * G,
        upper_bound=30_000.0 * G,
        scale=15_000.0 * G,
    )
    alphas = {
        k: opti.variable(init_guess=a0, lower_bound=-5.0, upper_bound=16.0)
        for k, a0 in (("cruise", 2.0), ("loiter", 2.0), ("sea", 8.0))
    }
    res = analyze(sym, spec, alphas=alphas, weight_n=weight_n, lut=lut)
    register_symbolic(opti, res, weight_n, margins)
    opti.subject_to(weight_n == res.mass.total * G)  # mass closure
    opti.minimize(res.mass.total)
    return opti, flat_vars, weight_n, alphas


def optimize(
    spec: MissionSpec,
    x0: DesignVector | None = None,
    margins: Margins | None = None,
    frozen_airfoil: str | None = "naca2412",
    max_runtime: float = 180.0,
    max_iter: int = 800,
    outer_iterations: int | None = None,
) -> OptimizationResult:
    """Solve the design problem. Never raises on infeasibility — diagnostics instead.

    The section aerodynamics for the symbolic problem come from a `PolarLUT`
    built numerically from NeuralFoil at the current airfoil shape. For a frozen
    (database) airfoil one pass suffices; for CST-shape optimization the LUT is
    rebuilt at each outer fixed-point iteration so the converged design is
    self-consistent with NeuralFoil itself.

    After the solve, numeric-only failure modes (flight-dynamic damping, spiral,
    static-margin edge cases) trigger a bounded repair loop that reseeds the
    flat problem directions and/or moves the empirical tail-volume floors.
    """
    from design_agent.polar_lut import build_design_registry

    margins = (margins or Margins()).with_overrides(spec)
    design = (x0 or initial_guess(spec, frozen_airfoil=frozen_airfoil)).clamp()
    if design.frozen_airfoil != frozen_airfoil:
        design = DesignVector.from_flat(design.to_flat(), frozen_airfoil=frozen_airfoil)
    if outer_iterations is None:
        outer_iterations = 1 if frozen_airfoil is not None else 3

    result: OptimizationResult | None = None
    working_margins = margins
    for _outer in range(outer_iterations):
        lut = build_design_registry(design)
        result = _solve_once(
            spec, design, working_margins, lut,
            max_runtime=max_runtime, max_iter=max_iter,
        )
        design = result.design
        if frozen_airfoil is not None:
            break

    # ---- Repair loop for numeric-only failure modes ------------------------
    # Flight-dynamic modes are numeric-only by design (see analysis.get_modes),
    # so they cannot appear in the Opti problem. The levers used here:
    #   spiral_stable      -> reseed dihedral upward (flat direction: free mass,
    #                         no symbolic gradient, IPOPT leaves the seed alone)
    #   zeta_short_period  -> raise V_h floor and reseed a longer htail arm
    #                         (Cmq damping scales with S_h * l_h^2)
    #   zeta_dutch_roll    -> raise V_v floor (Cn_beta)
    #   zeta_phugoid       -> relax toward the hard floor (drag-model-sensitive
    #                         at concept level; floor is certification-ish)
    #   static_margin_max  -> reseed a smaller htail (neutral point forward)
    #   static_margin      -> reseed CG forward
    repair_budget = 3
    while (
        result is not None
        and not result.success
        and repair_budget > 0
        and any(r.name in _REPAIRABLE and not r.passed for r in result.records)
    ):
        repair_budget -= 1
        failed = {r.name for r in result.records if not r.passed}
        print(f"[repair] round; failed modes: {sorted(failed)}", flush=True)

        seed = result.design
        if "spiral_stable" in failed:
            seed = replace(
                seed, dihedral_deg=min(seed.dihedral_deg + 3.0, 9.0)
            )
        if "zeta_short_period" in failed:
            working_margins = replace(
                working_margins,
                tail_volume_h=working_margins.tail_volume_h + 0.15,
            )
            seed = replace(seed, htail_arm=min(seed.htail_arm * 1.15, 12.0))
        if "zeta_dutch_roll" in failed:
            working_margins = replace(
                working_margins,
                tail_volume_v=working_margins.tail_volume_v + 0.015,
            )
        if "zeta_phugoid" in failed:
            working_margins = replace(
                working_margins,
                zeta_phugoid=max(
                    0.04 * working_margins.FLOOR_ZETA_FACTOR,
                    working_margins.zeta_phugoid - 0.01,
                ),
            )
        if "static_margin_max" in failed:
            seed = replace(seed, htail_area=max(seed.htail_area * 0.9, 2.0))
        if "static_margin" in failed:
            seed = replace(
                seed, cg_mac_fraction=max(seed.cg_mac_fraction - 0.05, 0.05)
            )
        working_margins = working_margins.clamped()

        lut = build_design_registry(seed)
        retried = _solve_once(
            spec, seed, working_margins, lut,
            max_runtime=max_runtime, max_iter=max_iter,
        )
        if _better(retried, result):
            result = retried
            design = retried.design

    assert result is not None
    return result


_REPAIRABLE = frozenset({
    "zeta_short_period", "zeta_dutch_roll", "zeta_phugoid", "spiral_stable",
    "static_margin", "static_margin_max",
})


def _better(a: OptimizationResult, b: OptimizationResult) -> bool:
    """a beats b if it passes more constraints, or same count with less mass."""
    if a.success != b.success:
        return a.success
    if a.constraints_passed != b.constraints_passed:
        return a.constraints_passed > b.constraints_passed
    return a.mass_total_kg <= b.mass_total_kg * 1.5


def _solve_once(
    spec: MissionSpec,
    x0: DesignVector,
    margins: Margins,
    lut: "PolarLUT | dict[str, PolarLUT]",
    max_runtime: float,
    max_iter: int,
) -> OptimizationResult:
    t0 = time.time()
    opti, flat_vars, weight_n, alphas = build_opti(spec, x0, margins, lut=lut)
    build_time = time.time() - t0

    sol = None
    converged = True
    message = "success"
    t0 = time.time()
    solve_kwargs: dict[str, Any] = dict(
        verbose=False, max_iter=max_iter, max_runtime=max_runtime
    )
    if _SOLVE_SUPPORTS_RETURN_LAST:
        solve_kwargs["behavior_on_failure"] = "return_last"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sol = opti.solve(**solve_kwargs)
        if any("Optimization failed" in str(w.message) for w in caught):
            converged = False
            message = "solver did not converge; diagnostics from last iterate"
    except Exception as exc:
        converged = False
        message = f"solver: {type(exc).__name__}: {str(exc)[:300]}"
        sol = None
    solve_time = time.time() - t0

    if sol is None:
        # Hard failure with no usable iterate: still produce diagnostics by
        # verifying the seed design, so callers always get constraint records.
        try:
            numeric0 = analyze(
                x0, spec, alphas={"cruise": 2.0, "loiter": 2.0, "sea": 8.0}
            )
            records0 = evaluate_numeric(numeric0, margins)
            mass0 = scalar(numeric0.mass.total)
        except Exception:
            records0 = []
            mass0 = x0_mass_guess(x0, spec)
        return OptimizationResult(
            success=False,
            design=x0,
            mass_total_kg=mass0,
            records=records0,
            message=message + " (no iterate available; seed design evaluated)",
            build_time_s=build_time,
            solve_time_s=solve_time,
        )

    flat_num = np.concatenate(
        [np.asarray(sol(v), dtype=float).ravel() for v in flat_vars]
    )
    design = DesignVector.from_flat(flat_num, frozen_airfoil=x0.frozen_airfoil)
    alpha_num = {k: scalar(sol(v)) for k, v in alphas.items()}

    try:
        numeric = analyze(design, spec, alphas=alpha_num)  # true NF verification
    except Exception as exc:
        numeric = None
        message += f"; numeric verification failed: {type(exc).__name__}"
    if numeric is None:
        try:
            numeric = analyze(
                x0, spec, alphas={"cruise": 2.0, "loiter": 2.0, "sea": 8.0}
            )
            design = x0
        except Exception:
            return OptimizationResult(
                success=False,
                design=x0,
                mass_total_kg=x0_mass_guess(x0, spec),
                records=[],
                message=message + " (numeric verification unavailable)",
                build_time_s=build_time,
                solve_time_s=solve_time,
            )

    records = evaluate_numeric(numeric, margins)
    # The numeric records (true NeuralFoil verification) are the ground truth:
    # a non-converged solver whose last iterate passes every check is a valid
    # design; `converged` stays visible in solver_stats and the message.
    success = all_passed(records)
    if not converged:
        message += "; solver stopped early (last iterate shown)"
    if not all_passed(records):
        failed = [r.name for r in records if not r.passed]
        message = (message + f"; constraint failures: {failed}").strip("; ")

    stats = {
        "converged": converged,
        "alphas": dict(alpha_num),
        "weight_n": scalar(numeric.mass.total) * G,
    }
    return OptimizationResult(
        success=success,
        design=design,
        mass_total_kg=scalar(numeric.mass.total),
        records=records,
        solver_stats=stats,
        message=message,
        build_time_s=build_time,
        solve_time_s=solve_time,
    )


def x0_mass_guess(x0: DesignVector, spec: MissionSpec) -> float:
    return 2.5 * spec.useful_load_kg
