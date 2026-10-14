"""End-to-end: natural-language mission -> parse -> optimize -> verified design.

Runs the full pipeline with the rule-based parser (no LLM) and a frozen
airfoil. Slow (minutes): set DESIGN_AGENT_SKIP_E2E=1 to skip on quick runs.
"""

import os

import pytest

from design_agent.agent_loop import AgentConfig, DesignAgent

MISSION = (
    "Design a 4-seat piston trainer. Payload 340 kg plus 2 crew. "
    "Range 800 km, cruise 60 m/s at 3000 m altitude. "
    "Takeoff distance under 600 m, stall speed below 30 m/s, "
    "20 minute reserve, climb at least 5 m/s."
)


@pytest.mark.skipif(
    os.environ.get("DESIGN_AGENT_SKIP_E2E"), reason="e2e disabled by env"
)
def test_agent_end_to_end_rule_based():
    agent = DesignAgent(
        AgentConfig(frozen_airfoil="naca2412", max_runtime=180.0, restarts=0)
    )
    report = agent.design_from_text(MISSION)

    # The rule-based parser must recover the key mission numbers.
    assert report.spec.payload_mass_kg == pytest.approx(340.0)
    assert report.spec.crew_count == 1  # rule-based parser: 1 crew for manned
    assert report.spec.range_m == pytest.approx(800e3)
    assert report.spec.cruise_speed_mps == pytest.approx(60.0)
    assert report.spec.max_takeoff_distance_m == pytest.approx(600.0)

    # The optimizer+verification pipeline must satisfy the catalog.
    assert report.result.constraints_total >= 12
    assert report.result.mass_total_kg > report.spec.useful_load_kg
    if not report.success:
        failed = [
            (r.name, r.value, r.limit)
            for r in report.result.records
            if not r.passed
        ]
        pytest.fail(f"agent left constraints unresolved: {failed}")

    # Audit trail must narrate parse + optimize at minimum.
    phases = [s.phase for s in report.steps]
    assert "parse" in phases and "optimize" in phases
