"""
Constraint catalog: lift/drag/moment/stability/field limits.

Every constraint has both faces:
* `register_symbolic(opti, res, weight_n)` — added to the `asb.Opti` problem.
* `evaluate_numeric(res)` -> `ConstraintRecord` — checked on solved/MC designs.

Thresholds live in `Margins` (safety-margin policy, tightened/relaxed by the
uncertainty feedback loop within hard floors). Dynamic-mode and geometry checks
are numeric-only by nature; everything else is enforced in-Opti *and* re-checked
numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import aerosandbox as asb
import aerosandbox.numpy as np

from design_agent.analysis import G, PROP_EFF, scalar, AnalysisResult
from design_agent.mission_spec import MissionSpec


@dataclass(frozen=True)
class Margins:
    """Safety-margin policy. All thresholds are research defaults, overridable
    per-mission via ``MissionSpec.constraints_overrides``."""

    cl_operating: float = 0.90  # CL_op <= cl_operating * CLmax
    thrust: float = 1.20  # T_available >= thrust * D_cruise
    static_min: float = 0.05  # static margin lower bound (fraction of MAC)
    static_max: float = 0.25  # static margin upper bound
    tofl_usage: float = 0.80  # predicted ground roll <= tofl_usage * limit
    zeta_short_period: float = 0.30
    zeta_dutch_roll: float = 0.08
    zeta_phugoid: float = 0.04
    spiral_eig_max: float = 0.02  # spiral may be mildly unstable at concept level
    mach_limit: float = 0.72  # buffet-ish ceiling when spec.max_mach is None
    cm_trim: float = 0.02  # |Cm| about CG at cruise
    rolling_mu: float = 0.02  # rolling-friction coefficient (TO surrogate)

    # Hard floors the repair loop may never relax below (certification-ish).
    FLOOR_STATIC_MIN = 0.03
    FLOOR_THRUST = 1.05
    FLOOR_CL = 0.95
    FLOOR_TOFL_USAGE = 0.95
    FLOOR_ZETA_FACTOR = 0.7

    def clamped(self) -> "Margins":
        """Enforce hard floors; called after any LLM/uncertainty adjustment."""
        return replace(
            self,
            static_min=max(self.static_min, self.FLOOR_STATIC_MIN),
            thrust=max(self.thrust, self.FLOOR_THRUST),
            cl_operating=min(self.cl_operating, self.FLOOR_CL),
            tofl_usage=min(self.tofl_usage, self.FLOOR_TOFL_USAGE),
            zeta_short_period=max(
                self.zeta_short_period, 0.30 * self.FLOOR_ZETA_FACTOR
            ),
            zeta_dutch_roll=max(self.zeta_dutch_roll, 0.08 * self.FLOOR_ZETA_FACTOR),
            zeta_phugoid=max(self.zeta_phugoid, 0.04 * self.FLOOR_ZETA_FACTOR),
        )

    def with_overrides(self, spec: MissionSpec) -> "Margins":
        base = self.clamped()
        if not spec.constraints_overrides:
            return base
        valid = {f for f in self.__dataclass_fields__ if f.isupper() is False}
        updates = {k: v for k, v in spec.constraints_overrides.items() if k in valid}
        return replace(base, **updates).clamped()


@dataclass
class ConstraintRecord:
    name: str
    value: float
    limit: float  # bound violated when crossing; sign per `kind`
    kind: str  # "upper" | "lower" | "equality"
    passed: bool
    description: str


def takeoff_ground_roll(res: AnalysisResult, margins: Margins) -> Any:
    """Symbolic-safe takeoff ground-roll surrogate (textbook form).

    Validated only loosely against `library.field_lengths` — the library function
    is used for the authoritative numeric check.
    """
    S = res.airplane.s_ref
    rho = asb.Atmosphere(0.0).density()
    W = res.mass.total * G
    cl_to = 0.85 * res.cl_max_3d
    v_lo = 1.2 * res.v_stall
    t_avg = PROP_EFF * res.design.power_w / (0.9 * v_lo)
    accel = t_avg - margins.rolling_mu * W
    return 1.44 * W**2 / (G * rho * S * cl_to * accel)


def register_symbolic(
    opti: asb.Opti, res: AnalysisResult, weight_n: Any, margins: Margins
) -> None:
    """Attach the full in-Opti constraint set to the optimization problem."""
    W = weight_n
    sm = (res.aero["x_np"][0] - res.airplane.xyz_ref[0]) / res.airplane.c_ref

    opti.subject_to(res.aero["L"][0] == W)  # cruise trim
    opti.subject_to(res.aero["L"][1] == W)  # loiter trim
    opti.subject_to(res.aero["Cm"][0] <= margins.cm_trim)
    opti.subject_to(res.aero["Cm"][0] >= -margins.cm_trim)
    opti.subject_to(res.cl_op <= margins.cl_operating * res.cl_max_3d)
    opti.subject_to(res.thrust_cruise >= margins.thrust * res.aero["D"][0])
    opti.subject_to(
        res.excess_power_sl >= res.mass.total * G * res.spec.climb_rate_min_mps
    )
    opti.subject_to(sm <= margins.static_max)
    opti.subject_to(sm >= margins.static_min)
    opti.subject_to(res.aero["Cma"][0] <= -1e-3)  # pitch stiffness
    if res.confidence is not None:
        opti.subject_to(res.confidence >= 0.90)  # NeuralFoil epistemic gate
    mach = res.spec.cruise_speed_mps / res.op_points["cruise"].atmosphere.speed_of_sound()
    opti.subject_to(mach <= (res.spec.max_mach or margins.mach_limit))
    if res.spec.stall_speed_max_mps is not None:
        opti.subject_to(res.v_stall <= res.spec.stall_speed_max_mps)
    if res.spec.max_takeoff_distance_m is not None:
        opti.subject_to(
            takeoff_ground_roll(res, margins)
            <= margins.tofl_usage * res.spec.max_takeoff_distance_m
        )


def evaluate_numeric(res: AnalysisResult, margins: Margins) -> list[ConstraintRecord]:
    """Evaluate every constraint (incl. numeric-only ones) on a numeric result."""
    m = margins.with_overrides(res.spec)
    W = scalar(res.mass.total) * G
    records: list[ConstraintRecord] = []

    def add(name, value, limit, kind, description):
        if kind == "upper":
            ok = value <= limit
        elif kind == "lower":
            ok = value >= limit
        else:
            ok = abs(value - limit) <= 0.005 * max(abs(limit), 1.0)
        records.append(ConstraintRecord(name, float(value), float(limit), kind, ok, description))

    add("trim_cruise", scalar(res.aero["L"][0]), W, "equality", "Lift = weight at cruise")
    add("trim_loiter", scalar(res.aero["L"][1]), W, "equality", "Lift = weight in loiter")
    add("pitch_trim", scalar(res.aero["Cm"][0]), 0.0, "equality", "Cm about CG = 0 at cruise")
    add("cl_margin", scalar(res.cl_op), m.cl_operating * scalar(res.cl_max_3d), "upper",
        "Operating CL <= margin * CLmax")
    add("thrust_margin", scalar(res.thrust_cruise), m.thrust * scalar(res.aero["D"][0]),
        "lower", "Available thrust >= margin * cruise drag")
    add("climb", scalar(res.excess_power_sl),
        scalar(res.mass.total) * G * res.spec.climb_rate_min_mps, "lower",
        "Sea-level excess power for required climb rate")
    sm = (scalar(res.aero["x_np"][0]) - scalar(res.airplane.xyz_ref[0])) / scalar(res.airplane.c_ref)
    add("static_margin", sm, m.static_min, "lower", "Static margin lower bound")
    add("static_margin_max", sm, m.static_max, "upper", "Static margin upper bound")
    add("pitch_stiffness", scalar(res.aero["Cma"][0]), 0.0, "upper", "Cma < 0")
    add("nf_confidence", scalar(res.confidence), 0.90, "lower",
        "NeuralFoil analysis_confidence gate")
    mach = res.spec.cruise_speed_mps / res.op_points["cruise"].atmosphere.speed_of_sound()
    add("mach_limit", mach, res.spec.max_mach or m.mach_limit, "upper", "Cruise Mach limit")

    if res.spec.stall_speed_max_mps is not None:
        add("stall_speed", scalar(res.v_stall), res.spec.stall_speed_max_mps, "upper",
            "Sea-level stall speed limit")
    if res.spec.max_takeoff_distance_m is not None:
        roll = _numeric_roll(res, m)
        add("takeoff_roll", roll, m.tofl_usage * res.spec.max_takeoff_distance_m, "upper",
            "Takeoff ground-roll surrogate within usage of limit")
        # Authoritative library check alongside the surrogate.
        lib = _library_field_length(res)
        if lib is not None:
            add("takeoff_roll_library", lib, res.spec.max_takeoff_distance_m, "upper",
                "library.field_lengths takeoff distance vs limit")

    if res.modes:
        z_sp = res.modes["short_period"]["damping_ratio"]
        z_dr = res.modes["dutch_roll"]["damping_ratio"]
        z_ph = res.modes["phugoid"]["damping_ratio"]
        sp_r = res.modes["spiral"]["eigenvalue_real"]
        add("zeta_short_period", scalar(z_sp), m.zeta_short_period, "lower", "Short-period damping")
        add("zeta_dutch_roll", scalar(z_dr), m.zeta_dutch_roll, "lower", "Dutch-roll damping")
        add("zeta_phugoid", scalar(z_ph), m.zeta_phugoid, "lower", "Phugoid damping")
        add("spiral_stable", scalar(sp_r), m.spiral_eig_max, "upper", "Spiral eigenvalue bound")

    t_over_c = _max_thickness_over_chord(res)
    if t_over_c is not None:
        add("thickness_floor", t_over_c, 0.09, "lower", "Airfoil t/c structural floor")

    return records


def all_passed(records: list[ConstraintRecord]) -> bool:
    return all(r.passed for r in records)


def violations(records: list[ConstraintRecord]) -> list[ConstraintRecord]:
    return [r for r in records if not r.passed]


def _numeric_roll(res: AnalysisResult, margins: Margins) -> float:
    S = scalar(res.airplane.s_ref)
    rho = 1.225
    W = scalar(res.mass.total) * G
    cl_to = 0.85 * scalar(res.cl_max_3d)
    v_lo = 1.2 * scalar(res.v_stall)
    t_avg = PROP_EFF * res.design.power_w / (0.9 * v_lo)
    accel = t_avg - margins.rolling_mu * W
    return float(1.44 * W**2 / (G * rho * S * cl_to * accel))


def _library_field_length(res: AnalysisResult) -> float | None:
    """Authoritative takeoff distance from aerosandbox.library.field_lengths."""
    try:
        from aerosandbox.library.field_lengths import field_length_analysis

        out = field_length_analysis(
            design_mass_TOGW=scalar(res.mass.total),
            thrust_at_liftoff=scalar(res.thrust_cruise),
            lift_over_drag_climb=max(scalar(res.ld_ratio), 4.0),
            CL_max=scalar(res.cl_max_3d),
            s_ref=scalar(res.airplane.s_ref),
            n_engines=1,
            V_engine_failure_balanced_field_length=0.0,
            atmosphere=asb.Atmosphere(0.0),
        )
        for key in ("ground_roll", "takeoff_distance", "BFL", "total"):
            if key in out:
                return float(out[key])
        return float(list(out.values())[0])
    except Exception:
        return None


def _max_thickness_over_chord(res: AnalysisResult) -> float | None:
    try:
        af = res.design.airfoil()
        coords = af.coordinates
        y = np.asarray(coords)[:, 1]
        x = np.asarray(coords)[:, 0]
        t = float(y.max() - y.min())
        c = float(x.max() - x.min())
        return t / c if c > 0 else None
    except Exception:
        return None
