"""
Agent orchestration: natural-language mission in, verified aircraft design out.

The agent is a thin autonomy layer over the design tools:

    parse mission (LLM if provided, else rule-based) -> MissionSpec
    -> optimize (symbolic NLP + true-NeuralFoil verification + bound repair)
    -> multi-start retry from perturbed seeds if constraints remain failed
    -> constraint-driven diagnosis and plain-language mission advice

Every action is logged as an `AgentStep`, so the CLI and the report generator
can narrate *why* the agent did what it did — the audit trail is part of the
research artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as onp

from design_agent.constraints import Margins, violations
from design_agent.mission_spec import MissionSpec
from design_agent.optimizer import OptimizationResult, optimize
from design_agent.parameterization import DesignVector


# --------------------------------------------------------------------- parsing


class _NullLLMClient:
    """OpenAI-shaped client that always fails fast -> rule-based fallback.

    Guarantees the parse stage is deterministic and network-free when no LLM
    client is supplied (the default for tests and offline runs).
    """

    def __init__(self) -> None:
        self.chat = type("chat", (), {})()
        self.chat.completions = self  # create() called directly on completions

    def create(self, **kwargs):
        raise RuntimeError("no LLM client configured; using rule-based parser")


def parse_mission_text(text: str, llm: Any = None) -> MissionSpec:
    """Parse natural-language mission text via `nl_parser.parse_mission`.

    `llm` is an optional OpenAI-compatible client. Without one, a null client
    forces the deterministic rule-based fallback. Returns just the spec; the
    `ParseTrace` audit record stays available through `nl_parser` directly.
    """
    from design_agent.nl_parser import parse_mission

    client = llm if llm is not None else _NullLLMClient()
    spec, _trace = parse_mission(text, client=client)
    return spec


# ------------------------------------------------------------------- data types


@dataclass
class AgentStep:
    """One entry in the agent's audit trail."""

    phase: str  # "parse" | "optimize" | "restart" | "diagnose"
    detail: str
    elapsed_s: float


@dataclass
class AgentConfig:
    frozen_airfoil: str | None = "naca2412"
    max_runtime: float = 180.0  # per-solve IPOPT budget [s]
    max_iter: int = 400
    restarts: int = 2  # perturbed-seed retries if constraints remain failed
    perturb_sigma: float = 0.03
    seed: int = 0
    margins: Margins = field(default_factory=Margins)


@dataclass
class AgentReport:
    spec: MissionSpec
    result: OptimizationResult
    steps: list[AgentStep]
    advice: list[str]

    @property
    def success(self) -> bool:
        return self.result.success


# -------------------------------------------------------------------- diagnosis

_ADVICE = {
    "trim_cruise": "Cannot trim lift against weight at cruise — allow a larger "
                   "wing or relax payload/range.",
    "trim_loiter": "Cannot trim at loiter speed — check wing area and loiter "
                   "speed assumptions.",
    "pitch_trim": "Pitch trim out of tolerance — horizontal-tail volume or CG "
                  "range is limiting; check htail_area/htail_arm bounds.",
    "cl_margin": "Operating CL too close to CLmax — larger wing area or a "
                 "higher-lift airfoil.",
    "thrust_margin": "Insufficient thrust margin at cruise — more power.",
    "climb": "Sea-level excess power below the climb requirement — more power "
             "or less drag/mass.",
    "static_margin": "Static margin below the floor — larger horizontal tail, "
                     "longer tail arm, or forward CG.",
    "static_margin_max": "Static margin above the ceiling (over-stable) — "
                         "smaller tail or aft CG.",
    "pitch_stiffness": "Cma not negative — configuration is pitch-unstable.",
    "nf_confidence": "NeuralFoil confidence below gate — airfoil/conditions "
                     "outside the trusted envelope.",
    "mach_limit": "Cruise Mach too high for the airfoil (buffet onset).",
    "stall_speed": "Stall speed above the requirement — more wing area or "
                   "higher CLmax.",
    "takeoff_roll": "Takeoff ground roll too long — more power, more wing, or "
                    "higher CLmax.",
    "takeoff_roll_library": "Library takeoff distance exceeds the limit.",
    "zeta_short_period": "Short-period damping low — larger horizontal tail or "
                         "longer tail arm.",
    "zeta_dutch_roll": "Dutch-roll damping low — larger vertical tail.",
    "zeta_phugoid": "Phugoid damping low — reduce L/D sensitivity (drag rise).",
    "spiral_stable": "Spiral mode too unstable — more dihedral or larger "
                     "vertical tail.",
    "thickness_floor": "Airfoil thinner than the structural floor.",
}


def diagnose(result: OptimizationResult) -> list[str]:
    """Plain-language advice for every violated constraint."""
    out = []
    for r in violations(result.records):
        hint = _ADVICE.get(r.name, "no specific advice registered")
        out.append(f"{r.name}: value {r.value:.4g} vs {r.kind} limit "
                   f"{r.limit:.4g} — {hint}")
    if not out and not result.success:
        out.append(f"solver reported: {result.message}")
    return out


# ------------------------------------------------------------------------ agent


def _better(a: OptimizationResult, b: OptimizationResult) -> bool:
    """a strictly preferable to b: more constraints passed, then lighter."""
    return (a.constraints_passed, -a.mass_total_kg) > (
        b.constraints_passed, -b.mass_total_kg,
    )


class DesignAgent:
    """Autonomous mission-to-design loop with an auditable step log."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    # ---- entry points ----

    def design_from_text(self, text: str, llm: Any = None) -> AgentReport:
        t0 = time.time()
        steps: list[AgentStep] = []
        spec = parse_mission_text(text, llm=llm)
        steps.append(AgentStep(
            "parse",
            f"mission parsed: role={spec.role!r}, payload={spec.payload_mass_kg:.0f} kg",
            time.time() - t0,
        ))
        return self.design_from_spec(spec, steps=steps)

    def design_from_spec(
        self, spec: MissionSpec, steps: list[AgentStep] | None = None
    ) -> AgentReport:
        steps = list(steps or [])
        cfg = self.config
        t0 = time.time()

        result = optimize(
            spec,
            margins=cfg.margins,
            frozen_airfoil=cfg.frozen_airfoil,
            max_runtime=cfg.max_runtime,
            max_iter=cfg.max_iter,
        )
        steps.append(AgentStep(
            "optimize",
            f"{result.constraints_passed}/{result.constraints_total} constraints "
            f"passed, mass {result.mass_total_kg:.0f} kg",
            time.time() - t0,
        ))

        # Multi-start: retry from perturbed seeds around the best design.
        rng = onp.random.default_rng(cfg.seed)
        for k in range(cfg.restarts):
            if result.success:
                break
            t0 = time.time()
            x0 = result.design.perturb(rng, geometry_sigma=cfg.perturb_sigma)
            retry = optimize(
                spec,
                x0=x0,
                margins=cfg.margins,
                frozen_airfoil=cfg.frozen_airfoil,
                max_runtime=cfg.max_runtime,
                max_iter=cfg.max_iter,
            )
            improved = _better(retry, result)
            if improved:
                result = retry
            steps.append(AgentStep(
                "restart",
                f"restart {k + 1}: {retry.constraints_passed}/"
                f"{retry.constraints_total} passed, mass "
                f"{retry.mass_total_kg:.0f} kg ({'kept' if improved else 'discarded'})",
                time.time() - t0,
            ))

        advice = diagnose(result)
        steps.append(AgentStep(
            "diagnose",
            "all constraints satisfied" if result.success
            else f"{len(advice)} unresolved constraint(s)",
            0.0,
        ))
        return AgentReport(spec=spec, result=result, steps=steps, advice=advice)
