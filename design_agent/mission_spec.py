"""
Structured mission-requirement specification.

`MissionSpec` is the single source of truth that the rest of the pipeline consumes:
the LLM parser produces it, the optimizer reads it, and the report archives it.
All quantities are SI base units (kg, m, s) — the AeroSandbox ecosystem convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class MissionSpec(BaseModel):
    """
    A validated, SI-unit description of what the aircraft must do.

    Required fields are the minimum needed to size a conventional fixed-wing
    aircraft: a payload, a range, and a cruise condition.
    """

    role: str = "unspecified aircraft"
    """Free-text description of the aircraft's job, e.g. '4-seat trainer'."""

    config: str = "conventional"
    """Configuration family. Only 'conventional' is supported by the MVP pipeline."""

    propulsion: str = "piston"
    """Propulsion family. 'piston' is the only fully wired option; reserved for extension."""

    payload_mass_kg: float = Field(ge=0.0)
    """Mission payload (passengers + baggage + mission equipment), excluding crew. [kg]"""

    crew_count: int = Field(default=1, ge=0)
    """Number of crew members."""

    crew_mass_kg: float = Field(default=90.0, gt=0)
    """Assumed mass per crew member (person + gear). [kg]"""

    range_m: float = Field(gt=0)
    """Design range. [m]"""

    cruise_speed_mps: float = Field(gt=0)
    """Design cruise true airspeed. [m/s]"""

    cruise_altitude_m: float = Field(ge=0.0, le=20_000.0)
    """Design cruise altitude. [m]"""

    max_takeoff_distance_m: float | None = Field(default=None, gt=0)
    """Maximum allowed ground-roll + obstacle-clearance distance; None = no explicit limit. [m]"""

    stall_speed_max_mps: float | None = Field(default=None, gt=0)
    """Maximum allowed stall speed; None = no explicit limit. [m/s]"""

    endurance_reserve_min: float = Field(default=45.0, ge=0.0)
    """Fuel reserve endurance at cruise conditions. [min]"""

    climb_rate_min_mps: float = Field(default=5.0, ge=0.0)
    """Minimum required sea-level rate of climb. [m/s]"""

    max_mach: float | None = Field(default=None, gt=0.0)
    """Optional cruise Mach limit; None = limit comes from compressibility physics."""

    constraints_overrides: dict[str, float] = Field(default_factory=dict)
    """Optional per-constraint threshold overrides, keyed by constraint name."""

    @model_validator(mode="after")
    def _check_speeds(self) -> "MissionSpec":
        if (
            self.stall_speed_max_mps is not None
            and self.stall_speed_max_mps >= self.cruise_speed_mps
        ):
            raise ValueError(
                "stall_speed_max_mps must be less than cruise_speed_mps."
            )
        return self

    @property
    def useful_load_kg(self) -> float:
        """Crew + payload mass. [kg]"""
        return self.payload_mass_kg + self.crew_count * self.crew_mass_kg

    def to_prompt_dict(self) -> dict[str, object]:
        """Compact, human-readable rendering for LLM prompts."""
        return {
            "role": self.role,
            "config": self.config,
            "propulsion": self.propulsion,
            "payload_mass_kg": self.payload_mass_kg,
            "crew": f"{self.crew_count} x {self.crew_mass_kg} kg",
            "range_km": self.range_m / 1e3,
            "cruise_speed_m_per_s": self.cruise_speed_mps,
            "cruise_altitude_m": self.cruise_altitude_m,
            "max_takeoff_distance_m": self.max_takeoff_distance_m,
            "stall_speed_max_m_per_s": self.stall_speed_max_mps,
            "endurance_reserve_min": self.endurance_reserve_min,
            "climb_rate_min_m_per_s": self.climb_rate_min_mps,
            "max_mach": self.max_mach,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "MissionSpec":
        return cls.model_validate(json.loads(Path(path).read_text()))

    # ---- Canned examples used by tests, demos, and few-shot prompting ----

    @classmethod
    def example_trainer(cls) -> "MissionSpec":
        """4-seat piston trainer; the 'small GA' reference case."""
        return cls(
            role="4-seat piston trainer",
            payload_mass_kg=340.0,  # 3 pax + bags
            crew_count=1,
            range_m=800e3,
            cruise_speed_mps=60.0,
            cruise_altitude_m=3_000.0,
            max_takeoff_distance_m=600.0,
            stall_speed_max_mps=30.0,
            endurance_reserve_min=20.0,
            climb_rate_min_mps=5.0,
        )

    @classmethod
    def example_feeder_transport(cls) -> "MissionSpec":
        """9-seat commuter; the 'light transport' reference case."""
        return cls(
            role="9-seat feeder transport",
            payload_mass_kg=900.0,  # 8 pax + bags
            crew_count=1,
            crew_mass_kg=90.0,
            range_m=500e3,
            cruise_speed_mps=90.0,
            cruise_altitude_m=4_500.0,
            max_takeoff_distance_m=1_000.0,
            stall_speed_max_mps=38.0,
            endurance_reserve_min=30.0,
            climb_rate_min_mps=6.0,
        )

    @classmethod
    def example_surveillance(cls) -> "MissionSpec":
        """Single-seat surveillance aircraft optimized for long loiter."""
        return cls(
            role="single-seat surveillance aircraft",
            payload_mass_kg=120.0,  # sensors
            crew_count=1,
            range_m=600e3,
            cruise_speed_mps=55.0,
            cruise_altitude_m=2_500.0,
            max_takeoff_distance_m=500.0,
            stall_speed_max_mps=28.0,
            endurance_reserve_min=60.0,
            climb_rate_min_mps=4.0,
        )
