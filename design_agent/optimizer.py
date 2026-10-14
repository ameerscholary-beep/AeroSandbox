"""
Optimization problem assembly and solve.

Builds one `asb.Opti` problem per candidate design: symbolic design vector +
trim AoAs + the TOGW closure variable, AeroSandbox physics traced symbolically,
constraint catalog registered, objective = gross mass. On solver failure the
last IPOPT iterate is still packaged (with diagnostics) so the agent loop can
see *why* it failed instead of getting an exception.
"""

from __future__ import annotations

import time
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
    lut: "PolarLUT | None" = None,
) -> tuple[asb.Opti, list[Any], Any, Any]:
    """Create the Opti problem. Returns (opti, flat_vars, weight_n, symbolic_result)."""
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
    max_iter: int = 400,
    outer_iterations: int | None = None,
) -> OptimizationResult:
    """Solve the design problem. Never raises on infeasibility — diagnostics instead.

    The section aerodynamics for the symbolic problem come from a `PolarLUT`
    built numerically from NeuralFoil at the current airfoil shape. For a frozen
    (database) airfoil one pass suffices; for CST-shape optimization the LUT is
    rebuilt at each outer fixed-point iteration so the converged design is
    self-consistent with NeuralFoil itself.
    """
    from design_agent.polar_lut import PolarLUT

    margins = (margins or Margins()).with_overrides(spec)
    design = (x0 or initial_guess(spec, frozen_airfoil=frozen_airfoil)).clamp()
    if design.frozen_airfoil != frozen_airfoil:
        design = DesignVector.from_flat(design.to_flat(), frozen_airfoil=frozen_airfoil)
    if outer_iterations is None:
        outer_iterations = 1 if frozen_airfoil is not None else 3

    result: OptimizationResult | None = None
    working_margins = margins
    mode_repair_budget = 2
    for _outer in range(outer_iterations):
        lut = PolarLUT(design.airfoil())
        result = _solve_once(
            spec, design, working_margins, lut,
            max_runtime=max_runtime, max_iter=max_iter,
        )
        design = result.design
        if frozen_airfoil is not None:
            break
    # Dynamic-mode feedback: damping is checked numerically (modes are numeric-only
    # by design); if the optimum is lightly damped, stiffen the static-margin
    # requirement and re-solve. Bounded; floors still apply via clamped margins.
    while (
        result is not None
        and not result.success
        and mode_repair_budget > 0
        and any(r.name in ("zeta_short_period", "zeta_dutch_roll") and not r.passed
                for r in result.records)
    ):
        mode_repair_budget -= 1
        working_margins = replace(
            working_margins, static_min=max(working_margins.static_min, 0.12)
        ).clamped()
        lut = PolarLUT(design.airfoil())
        retried = _solve_once(
            spec, design, working_margins, lut,
            max_runtime=max_runtime, max_iter=max_iter,
        )
        if retried.mass_total_kg <= result.mass_total_kg * 1.5:
            result = retried
            design = retried.design
    assert result is not None
    return result


def _solve_once(
    spec: MissionSpec,
    x0: DesignVector,
    margins: Margins,
    lut: "PolarLUT",
    max_runtime: float,
    max_iter: int,
) -> OptimizationResult:
    t0 = time.time()
    opti, flat_vars, weight_n, alphas = build_opti(spec, x0, margins, lut=lut)
    build_time = time.time() - t0

    sol = None
    message = "success"
    t0 = time.time()
    try:
        sol = opti.solve(verbose=False, max_iter=max_iter, max_runtime=max_runtime)
    except Exception as exc:
        message = f"solver: {type(exc).__name__}: {str(exc)[:300]}"
        try:
            sol = opti.debug(suppress_errors=True)
        except Exception:
            sol = None
    solve_time = time.time() - t0

    if sol is None:
        return OptimizationResult(
            success=False,
            design=x0,
            mass_total_kg=x0_mass_guess(x0, spec),
            records=[],
            message=message + " (no iterate available)",
            build_time_s=build_time,
            solve_time_s=solve_time,
        )

    flat_num = np.concatenate(
        [np.asarray(sol(v), dtype=float).ravel() for v in flat_vars]
    )
    design = DesignVector.from_flat(flat_num, frozen_airfoil=x0.frozen_airfoil)
    alpha_num = {k: scalar(sol(v)) for k, v in alphas.items()}
    numeric = analyze(design, spec, alphas=alpha_num)  # true NF, fixed-point mass
    records = evaluate_numeric(numeric, margins)
    success = all_passed(records)
    if not success:
        failed = [r.name for r in records if not r.passed]
        message = (message + f"; constraint failures: {failed}").strip("; ")

    stats = {
        "iterations": getattr(sol, "stats", {}),
        "solver_message": message,
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
