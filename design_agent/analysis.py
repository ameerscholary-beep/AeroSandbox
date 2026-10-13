"""
Physics wrapper around AeroSandbox + NeuralFoil.

One `analyze()` call produces everything the optimizer and the verifier need:
Raymer statistical mass breakdown, vectorized AeroBuildup forces with stability
derivatives, NeuralFoil CLmax/confidence sweeps, and propulsion bookkeeping.

The same code runs numerically (floats) and symbolically (CasADi via `asb.Opti`
variables) because it is written entirely in `aerosandbox.numpy` — except dynamic
mode estimation (`get_modes`), which is numeric-only by design.

Mass closure
------------
Structural (empty) mass scales with takeoff gross weight (TOGW) via the Raymer
relations, while TOGW = useful load + fuel + empty mass. Two closures are offered:

* symbolic path — the optimizer owns an explicit ``weight_n`` variable; Raymer
  functions are traced against it and a closure constraint
  ``weight_n == mass.total * G`` is registered by the optimizer.
* numeric path — a short fixed-point iteration (Raymer's TOGW sensitivity is
  weak) converges empty mass before the result is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.library.weights import raymer_general_aviation_weights as raymer

from design_agent.mission_spec import MissionSpec
from design_agent.parameterization import DesignVector

G = 9.81
N_ULT = 5.7  # ultimate load factor (~+4.4 gust*1.5 for utility category)
PROP_EFF = 0.80  # propeller efficiency (cruise)
ENGINE_INSTALLED_KG_PER_KW = 1.3  # piston engine + propeller install, dry
BSFC_KG_PER_J = 0.28 / 3.6e6  # 0.28 kg/kWh shaft specific fuel consumption
FUEL_DENSITY = 800.0  # kg/m^3
CLMAX_3D_FACTOR = 0.95  # finite-wing CLmax derate from 2D airfoil data
SWEEP_N_ALPHA = 26  # NeuralFoil AoA sweep resolution for CLmax
_MODE_AERO_KEYS = {"CL", "CD", "a", "b", "p", "q", "r"}


def is_symbolic(value: Any) -> bool:
    return bool(np.determine_type.is_casadi_type(value))


def scalar(value: Any) -> float:
    """Extract a Python float from a numeric or 0/1-d symbolic value."""
    return float(np.asarray(value).ravel()[0])


@dataclass
class MassBreakdown:
    """Mass bookkeeping, all kg. `total = useful + fuel + empty`."""

    wing: Any
    fuselage: Any
    hstab: Any
    vstab: Any
    engine: Any
    gear: Any
    flight_controls: Any
    fuel_system: Any
    avionics: Any
    electrical: Any
    furnishings: Any
    empty: Any
    fuel: Any
    useful: float
    total: Any

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AnalysisResult:
    """Full analysis of one design point (numeric or symbolic)."""

    design: DesignVector
    airplane: asb.Airplane
    spec: MissionSpec
    mass: MassBreakdown
    aero: dict[str, Any]  # vectorized run_with_stability_derivatives output
    op_points: dict[str, asb.OperatingPoint]
    cl_max_2d: Any
    cl_max_3d: Any
    cl_op: Any  # cruise CL
    ld_ratio: Any  # cruise L/D
    v_stall: Any  # m/s, sea level, at CLmax
    thrust_cruise: Any  # N available at cruise
    excess_power_sl: Any  # W available minus required, sea level
    confidence: Any  # min NeuralFoil analysis_confidence over the CLmax sweep
    modes: dict[str, dict[str, Any]] | None = None  # numeric only

    @property
    def is_numeric(self) -> bool:
        return not is_symbolic(self.mass.total)


def analyze(
    design: DesignVector,
    spec: MissionSpec,
    alphas: dict[str, Any] | None = None,
    weight_n: Any | None = None,
    compute_modes: bool | None = None,
) -> AnalysisResult:
    """
    Analyze one design point.

    Parameters
    ----------
    design, spec
        Design point and mission requirements.
    alphas : dict[str, Any] | None
        Trim angle-of-attack per operating point (e.g. ``{"cruise": opti_var}``).
        Missing points default to sensible values. The optimizer passes variables.
    weight_n : Any | None
        Symbolic design gross weight [N] for the Raymer closure. If None, a
        numeric fixed-point closure is performed instead.
    compute_modes : bool | None
        Estimate dynamic modes (numeric only). Auto-decided if None.
    """
    alphas = alphas or {}
    airplane = design.to_airplane()

    # ---- Operating points ----
    op_cruise = asb.OperatingPoint(
        velocity=spec.cruise_speed_mps,
        alpha=alphas.get("cruise", 2.0),
        atmosphere=asb.Atmosphere(spec.cruise_altitude_m),
    )
    op_loiter = asb.OperatingPoint(
        velocity=0.75 * spec.cruise_speed_mps,
        alpha=alphas.get("loiter", 2.0),
        atmosphere=asb.Atmosphere(spec.cruise_altitude_m),
    )
    v_sl = 1.3 * (spec.stall_speed_max_mps or 35.0)
    op_sea = asb.OperatingPoint(
        velocity=v_sl,
        alpha=alphas.get("sea", 8.0),
        atmosphere=asb.Atmosphere(0.0),
    )
    op_points = {"cruise": op_cruise, "loiter": op_loiter, "sea": op_sea}

    # ---- Aerodynamics: one vectorized call ----
    op_all = asb.OperatingPoint(
        velocity=np.stack(
            [op_cruise.velocity, op_loiter.velocity, op_sea.velocity]
        ),
        alpha=np.stack([op_cruise.alpha, op_loiter.alpha, op_sea.alpha]),
        atmosphere=asb.Atmosphere(
            altitude=np.stack(
                [
                    op_cruise.atmosphere.altitude,
                    op_loiter.atmosphere.altitude,
                    op_sea.atmosphere.altitude,
                ]
            )
        ),
    )
    aero = asb.AeroBuildup(airplane=airplane, op_point=op_all).run_with_stability_derivatives()

    d_cruise = aero["D"][0]
    d_sea = aero["D"][2]
    ld_ratio = aero["L"][0] / d_cruise

    # ---- Mass ----
    mass = _mass_model(design, spec, airplane, aero, op_cruise, ld_ratio, weight_n)

    # ---- NeuralFoil CLmax + confidence sweep (root section) ----
    atmo_c = op_cruise.atmosphere
    re_root = (
        atmo_c.density() * spec.cruise_speed_mps * design.root_chord
        / atmo_c.dynamic_viscosity()
    )
    mach = spec.cruise_speed_mps / atmo_c.speed_of_sound()
    sweep = design.airfoil().get_aero_from_neuralfoil(
        alpha=np.linspace(-5, 20, SWEEP_N_ALPHA),
        Re=re_root,
        mach=mach,
        model_size="large",
    )
    cl_max_2d = np.max(sweep["CL"])
    confidence = np.min(sweep["analysis_confidence"])
    cl_max_3d = CLMAX_3D_FACTOR * cl_max_2d

    v_stall = np.sqrt(
        2 * mass.total * G
        / (asb.Atmosphere(0.0).density() * airplane.s_ref * cl_max_3d)
    )

    # ---- Propulsion ----
    thrust_cruise = PROP_EFF * design.power_w / spec.cruise_speed_mps
    excess_power_sl = PROP_EFF * design.power_w - d_sea * v_sl

    # ---- Dynamic modes (numeric only) ----
    if compute_modes is None:
        compute_modes = not is_symbolic(mass.total)
    modes = None
    if compute_modes:
        aero_for_modes = {
            k: scalar(v)
            for k, v in aero.items()
            if k in ("CL", "CD") or k[2:] in _MODE_AERO_KEYS
        }
        mp = asb.MassProperties(
            mass=scalar(mass.total),
            xyz_cg=[scalar(c) for c in airplane.xyz_ref],
            Ixx=0.08 * scalar(mass.total) * (design.span / 2) ** 2,
            Iyy=0.10 * scalar(mass.total) * design.fuselage_length ** 2,
            Izz=0.14 * scalar(mass.total) * design.fuselage_length ** 2,
        )
        try:
            modes = get_modes(
                airplane=airplane, op_point=op_cruise, mass_props=mp,
                aero=aero_for_modes,
            )
        except Exception:
            modes = None

    return AnalysisResult(
        design=design,
        airplane=airplane,
        spec=spec,
        mass=mass,
        aero=aero,
        op_points=op_points,
        cl_max_2d=cl_max_2d,
        cl_max_3d=cl_max_3d,
        cl_op=aero["CL"][0],
        ld_ratio=ld_ratio,
        v_stall=v_stall,
        thrust_cruise=thrust_cruise,
        excess_power_sl=excess_power_sl,
        confidence=confidence,
        modes=modes,
    )


def _empty_mass(design, airplane, aero, op_cruise, ld_ratio, togw_kg) -> Any:
    """Raymer GA statistical empty mass, traced against a (numeric or symbolic) TOGW."""
    fuse = airplane.fuselages[0]
    wing, hstab, vstab = airplane.wings
    m_wing = raymer.mass_wing(
        wing=wing, design_mass_TOGW=togw_kg, ultimate_load_factor=N_ULT,
        mass_fuel_in_wing=0.0, cruise_op_point=op_cruise,
    )
    m_fuse = raymer.mass_fuselage(
        fuselage=fuse, design_mass_TOGW=togw_kg, ultimate_load_factor=N_ULT,
        L_over_D=ld_ratio, cruise_op_point=op_cruise,
        wing_to_tail_distance=design.htail_arm, pressure_differential=0.0,
    )
    m_hstab = raymer.mass_hstab(
        hstab=hstab, design_mass_TOGW=togw_kg, ultimate_load_factor=N_ULT,
        cruise_op_point=op_cruise,
    )
    m_vstab = raymer.mass_vstab(
        vstab=vstab, design_mass_TOGW=togw_kg, ultimate_load_factor=N_ULT,
        cruise_op_point=op_cruise,
    )
    m_engine = raymer.mass_engines_installed(
        n_engines=1,
        mass_per_engine=ENGINE_INSTALLED_KG_PER_KW * design.power_w / 1e3,
    )
    m_gear = raymer.mass_main_landing_gear(
        main_gear_length=design.fuselage_radius * 1.5, design_mass_TOGW=togw_kg,
    ) + raymer.mass_nose_landing_gear(
        nose_gear_length=design.fuselage_radius * 1.5, design_mass_TOGW=togw_kg,
    )
    m_controls = raymer.mass_flight_controls(
        airplane=airplane, design_mass_TOGW=togw_kg, ultimate_load_factor=N_ULT,
        fuselage=fuse, main_wing=wing,
    )
    m_avionics = raymer.mass_avionics(mass_uninstalled_avionics=40.0)
    return {
        "wing": m_wing, "fuselage": m_fuse, "hstab": m_hstab, "vstab": m_vstab,
        "engine": m_engine, "gear": m_gear, "flight_controls": m_controls,
        "avionics": m_avionics,
    }


def _fuel_mass(design, spec, d_cruise, fuel_system_mass, electrical_mass,
               furnishings_mass, empty_so_far, useful_kg) -> Any:
    """Fuel for cruise + reserve with 10% unusable margin. Depends on drag."""
    p_shaft = d_cruise * spec.cruise_speed_mps / PROP_EFF
    t_mission = spec.range_m / spec.cruise_speed_mps + spec.endurance_reserve_min * 60.0
    return 1.1 * p_shaft * t_mission * BSFC_KG_PER_J


def _mass_model(design, spec, airplane, aero, op_cruise, ld_ratio, weight_n):
    """Assemble the MassBreakdown, handling symbolic vs numeric closure."""
    d_cruise = aero["D"][0]
    useful = spec.useful_load_kg

    def totals(togw_kg):
        parts = _empty_mass(design, airplane, aero, op_cruise, ld_ratio, togw_kg)
        m_fuel = _fuel_mass(
            design, spec, d_cruise, 0.0, 0.0, 0.0, 0.0, useful
        )
        m_fs = raymer.mass_fuel_system(
            fuel_volume=m_fuel / FUEL_DENSITY, n_tanks=2, n_engines=1,
        )
        m_elec = raymer.mass_electrical(
            fuel_system_mass=m_fs, avionics_mass=parts["avionics"]
        )
        m_furn = raymer.mass_furnishings(design_mass_TOGW=togw_kg)
        empty = sum(parts.values()) + m_fs + m_elec + m_furn
        return parts, m_fs, m_elec, m_furn, m_fuel, empty

    if weight_n is not None:
        # Symbolic closure: trace everything against the optimizer's weight variable.
        togw_kg = weight_n / G
        parts, m_fs, m_elec, m_furn, m_fuel, empty = totals(togw_kg)
        total = useful + m_fuel + empty
        return MassBreakdown(
            **parts, fuel_system=m_fs, electrical=m_elec, furnishings=m_furn,
            empty=empty, fuel=m_fuel, useful=useful, total=total,
        )

    # Numeric fixed-point closure (Raymer TOGW sensitivity is weak; 4 iterations).
    togw_kg = 2.5 * useful
    for _ in range(4):
        parts, m_fs, m_elec, m_furn, m_fuel, empty = totals(togw_kg)
        togw_kg = useful + scalar(m_fuel) + scalar(empty)
    total = useful + m_fuel + empty
    return MassBreakdown(
        **parts, fuel_system=m_fs, electrical=m_elec, furnishings=m_furn,
        empty=empty, fuel=m_fuel, useful=useful, total=total,
    )
