"""
Design-variable parameterization: the bridge between the optimizer and AeroSandbox geometry.

A `DesignVector` is a compact, bounded, well-conditioned description of a conventional
fixed-wing aircraft (wing planform + CST airfoil + empennage + fuselage + CG + power).
It converts to/from a flat array (for `Opti` variables) and builds an `asb.Airplane`
that works both numerically and symbolically (all math via `aerosandbox.numpy`).
"""

from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np

N_CST_WEIGHTS = 8
"""Number of Kulfan (CST) weights per airfoil surface."""


def _seed_airfoil_fields(
    values: dict, frozen_airfoil: str | None
) -> dict:
    """Fill the CST airfoil fields from a database seed when airfoil is frozen."""
    values["frozen_airfoil"] = frozen_airfoil
    if frozen_airfoil is not None and "lower_weights" not in values:
        seed = asb.KulfanAirfoil(name=frozen_airfoil)
        values["lower_weights"] = seed.lower_weights
        values["upper_weights"] = seed.upper_weights
        values["leading_edge_weight"] = float(seed.leading_edge_weight)
        values["te_thickness"] = float(seed.TE_thickness)
    return values


@dataclass(frozen=True)
class FieldSpec:
    """Bounds and scaling metadata for one design-variable block."""

    name: str
    lo: float
    hi: float
    scale: float
    log_transform: bool = False
    size: int = 1


@dataclass(frozen=True)
class DesignVector:
    """Bounded design description of a conventional fixed-wing aircraft.

    All fields are plain floats (or float arrays) for numeric use. The optimizer
    creates a parallel *symbolic* DesignVector by mapping each field through
    `Opti.variable`; `to_airplane` accepts either.
    """

    span: float
    root_chord: float
    taper: float
    sweep_qc_deg: float
    twist_tip_deg: float
    dihedral_deg: float

    lower_weights: np.ndarray  # (N_CST_WEIGHTS,)
    upper_weights: np.ndarray  # (N_CST_WEIGHTS,)
    leading_edge_weight: float
    te_thickness: float

    htail_area: float
    vtail_area: float
    htail_arm: float
    vtail_arm: float

    fuselage_length: float
    fuselage_radius: float

    cg_mac_fraction: float
    power_w: float

    frozen_airfoil: str | None = None
    """If set (e.g. 'naca2412'), the CST airfoil fields are ignored and this
    database airfoil is used instead (cheap mode)."""

    # ------------------------------------------------------------------ #
    # Field/bounds table
    # ------------------------------------------------------------------ #

    @staticmethod
    def _airfoil_specs() -> list[FieldSpec]:
        return [
            FieldSpec("lower_weights", -0.35, 0.35, 0.1, size=N_CST_WEIGHTS),
            FieldSpec("upper_weights", -0.30, 0.55, 0.1, size=N_CST_WEIGHTS),
            FieldSpec("leading_edge_weight", -0.10, 0.20, 0.03),
            FieldSpec("te_thickness", 0.0, 0.02, 0.003),
        ]

    @classmethod
    def field_specs(cls, frozen_airfoil: str | None = None) -> list[FieldSpec]:
        """Ordered variable blocks. Airfoil blocks are omitted when frozen."""
        base = [
            FieldSpec("span", 6.0, 22.0, 10.0),
            FieldSpec("root_chord", 0.8, 3.5, 1.5),
            FieldSpec("taper", 0.2, 1.0, 0.6),
            FieldSpec("sweep_qc_deg", 0.0, 25.0, 10.0),
            FieldSpec("twist_tip_deg", -5.0, 3.0, 1.0),
            FieldSpec("dihedral_deg", 0.0, 10.0, 3.0),
        ]
        if frozen_airfoil is None:
            base += cls._airfoil_specs()
        base += [
            FieldSpec("htail_area", 2.0, 25.0, 5.0),
            FieldSpec("vtail_area", 1.0, 15.0, 3.0),
            FieldSpec("htail_arm", 2.5, 12.0, 4.0),
            FieldSpec("vtail_arm", 2.5, 12.0, 4.0),
            FieldSpec("fuselage_length", 5.0, 20.0, 7.0),
            FieldSpec("fuselage_radius", 0.5, 2.0, 0.8),
            FieldSpec("cg_mac_fraction", 0.05, 0.40, 0.25),
            FieldSpec("power_w", 20e3, 800e3, 100e3, log_transform=True),
        ]
        return base

    # ------------------------------------------------------------------ #
    # Flat-array conversion (for Opti variables / LLM guesses / manifests)
    # ------------------------------------------------------------------ #

    def to_flat(self) -> np.ndarray:
        specs = self.field_specs(self.frozen_airfoil)
        return np.concatenate(
            [np.asarray(getattr(self, s.name), dtype=float).ravel() for s in specs]
        )

    @classmethod
    def from_flat(
        cls, arr: np.ndarray, frozen_airfoil: str | None = None
    ) -> "DesignVector":
        arr = np.asarray(arr, dtype=float).ravel()
        values: dict[str, np.ndarray | float] = {}
        i = 0
        for s in cls.field_specs(frozen_airfoil):
            block = arr[i : i + s.size]
            i += s.size
            values[s.name] = block if s.size > 1 else float(block[0])
        return cls(**_seed_airfoil_fields(values, frozen_airfoil))

    def clamp(self) -> "DesignVector":
        """Clip every field into its bounds (used for LLM-provided guesses)."""
        values: dict[str, np.ndarray | float | str | None] = {}
        for s in self.field_specs(self.frozen_airfoil):
            raw = np.asarray(getattr(self, s.name), dtype=float)
            clipped = np.clip(raw, s.lo, s.hi)
            values[s.name] = clipped if s.size > 1 else float(clipped.ravel()[0])
        return DesignVector(**_seed_airfoil_fields(values, self.frozen_airfoil))

    def perturb(
        self, rng: np.random.Generator, geometry_sigma: float = 0.01
    ) -> "DesignVector":
        """Return a copy with dimensional geometry scaled by N(1, geometry_sigma).

        Used by the Monte-Carlo verifier for manufacturing/geometry dispersion.
        Mass and atmosphere dispersion are handled in `uncertainty.py`.
        """
        flat = np.array(self.to_flat(), dtype=float)
        offset = 0
        dimensional = {"span", "root_chord", "htail_area", "vtail_area",
                       "fuselage_length", "fuselage_radius"}
        for s in self.field_specs(self.frozen_airfoil):
            block = flat[offset : offset + s.size]
            if s.name in dimensional:
                factor = float(rng.normal(1.0, geometry_sigma))
                flat[offset : offset + s.size] = block * factor
            offset += s.size
        return DesignVector.from_flat(flat, self.frozen_airfoil)

    # ------------------------------------------------------------------ #
    # Geometry construction
    # ------------------------------------------------------------------ #

    def wing_aero_numbers(self) -> tuple[object, object, object]:
        """(S [m^2], MAC [m], x_le_mac [m]) of the trapezoidal wing."""
        c_r = self.root_chord
        lam = self.taper
        S = self.span * c_r * (1 + lam) / 2
        mac = (2 / 3) * c_r * (1 + lam + lam**2) / (1 + lam)
        y_mac = self.span / 6 * (1 + 2 * lam) / (1 + lam)
        x_le_mac = y_mac * np.tand(self.sweep_qc_deg) - 0.25 * (mac - c_r)
        return S, mac, x_le_mac

    def airfoil(self) -> asb.Airfoil:
        if self.frozen_airfoil is not None:
            return asb.Airfoil(self.frozen_airfoil)
        return asb.KulfanAirfoil(
            name="agent_cst",
            lower_weights=self.lower_weights,
            upper_weights=self.upper_weights,
            leading_edge_weight=self.leading_edge_weight,
            TE_thickness=self.te_thickness,
        )

    def to_airplane(self, name: str = "agent_design") -> asb.Airplane:
        """Build the full `asb.Airplane`. Works with numeric or symbolic fields."""
        span = self.span
        c_r = self.root_chord
        lam = self.taper
        S, mac, x_le_mac = self.wing_aero_numbers()
        x_cg = x_le_mac + self.cg_mac_fraction * mac

        # ---- Main wing: 3 sections per semispan ----
        af = self.airfoil()
        etas = [0.0, 0.4, 1.0]
        wing_xsecs = []
        for eta in etas:
            chord = c_r * (1 - (1 - lam) * eta)
            x_le = eta * span / 2 * np.tand(self.sweep_qc_deg) - 0.25 * (chord - c_r)
            z_le = eta * span / 2 * np.tand(self.dihedral_deg)
            twist = self.twist_tip_deg * eta
            wing_xsecs.append(
                asb.WingXSec(xyz_le=[x_le, eta * span / 2, z_le],
                             chord=chord, twist=twist, airfoil=af)
            )
        wing = asb.Wing(name="Main Wing", symmetric=True, xsecs=wing_xsecs)

        # ---- Empennage ----
        htail = _build_tail_wing(
            name="Horizontal Stabilizer",
            area=self.htail_area, aspect_ratio=4.2, taper=0.5,
            x_le=self._tail_le_x(x_le_mac, mac, self.htail_arm, self.htail_area, 4.2, 0.5),
            airfoil_name="naca0010", twist=-2.0,
        )
        vtail = _build_tail_wing(
            name="Vertical Stabilizer",
            area=self.vtail_area, aspect_ratio=1.5, taper=0.5,
            x_le=self._tail_le_x(x_le_mac, mac, self.vtail_arm, self.vtail_area, 1.5, 0.5),
            airfoil_name="naca0010", twist=0.0, vertical=True,
        )

        # ---- Fuselage ----
        xi = np.cosspace(0, 1, 15)
        thickness = asb.Airfoil("dae51").local_thickness(x_over_c=xi)
        x_nose = x_le_mac - 0.2 * self.fuselage_length
        fuse = asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[x_nose + xi_i * self.fuselage_length, 0, 0],
                    radius=self.fuselage_radius * t_i,
                )
                for xi_i, t_i in zip(xi, thickness)
            ],
        )

        return asb.Airplane(
            name=name,
            xyz_ref=[x_cg, 0, 0],
            wings=[wing, htail, vtail],
            fuselages=[fuse],
            s_ref=S,
            c_ref=mac,
            b_ref=span,
        )

    def _tail_le_x(
        self, x_le_mac, mac, arm, area, ar, taper
    ) -> object:
        span_t = np.sqrt(ar * area)
        c_root = 2 * area / (span_t * (1 + taper))
        return x_le_mac + 0.25 * mac + arm - 0.25 * c_root


def _build_tail_wing(
    name: str,
    area: float,
    aspect_ratio: float,
    taper: float,
    x_le: object,
    airfoil_name: str,
    twist: float,
    vertical: bool = False,
) -> asb.Wing:
    span_t = np.sqrt(aspect_ratio * area)
    c_root = 2 * area / (span_t * (1 + taper))
    c_tip = taper * c_root
    af = asb.Airfoil(airfoil_name)
    if vertical:
        xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=twist, airfoil=af),
            asb.WingXSec(xyz_le=[(c_root - c_tip) * 0.5, 0, span_t],
                         chord=c_tip, twist=twist, airfoil=af),
        ]
        wing = asb.Wing(name=name, symmetric=False, xsecs=xsecs)
    else:
        xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=twist, airfoil=af),
            asb.WingXSec(xyz_le=[(c_root - c_tip) * 0.5, span_t, 0],
                         chord=c_tip, twist=twist, airfoil=af),
        ]
        wing = asb.Wing(name=name, symmetric=True, xsecs=xsecs)
    return wing.translate([x_le, 0, 0])


# --------------------------------------------------------------------------- #
# Initial sizing heuristic
# --------------------------------------------------------------------------- #


def initial_guess(spec, frozen_airfoil: str | None = None) -> DesignVector:
    """Textbook-trend initial design point, clamped into the valid box."""
    W_pay = spec.useful_load_kg * 9.81
    W_togw_guess = 2.6 * W_pay  # crude growth factor for small aircraft
    S = W_togw_guess / 800.0  # wing loading ~800 N/m^2
    ar = 8.0
    span = float(np.sqrt(S * ar))
    taper = 0.5
    root_chord = 2 * S / (span * (1 + taper))
    power = W_togw_guess * spec.cruise_speed_mps / (9.0 * 0.75)

    x = DesignVector(
        span=span,
        root_chord=root_chord,
        taper=taper,
        sweep_qc_deg=3.0,
        twist_tip_deg=0.0,
        dihedral_deg=3.0,
        lower_weights=np.zeros(N_CST_WEIGHTS),
        upper_weights=np.zeros(N_CST_WEIGHTS),
        leading_edge_weight=0.0,
        te_thickness=0.0,
        htail_area=0.22 * S,
        vtail_area=0.13 * S,
        htail_arm=0.45 * span,
        vtail_arm=0.42 * span,
        fuselage_length=max(6.0, 0.8 * span),
        fuselage_radius=1.0,
        cg_mac_fraction=0.25,
        power_w=power,
        frozen_airfoil=frozen_airfoil,
    )
    if frozen_airfoil is None:
        seed = asb.KulfanAirfoil(name="naca2412")
        x = DesignVector(
            **{
                **x.__dict__,
                "lower_weights": seed.lower_weights,
                "upper_weights": seed.upper_weights,
                "leading_edge_weight": float(seed.leading_edge_weight),
                "te_thickness": float(seed.TE_thickness),
            }
        )
    return x.clamp()
