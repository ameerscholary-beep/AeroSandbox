"""Markdown report generation: sections, tables, robustness, figures."""

import matplotlib

matplotlib.use("Agg")

from design_agent.agent_loop import AgentReport, AgentStep, diagnose
from design_agent.analysis import analyze, scalar
from design_agent.constraints import Margins, evaluate_numeric
from design_agent.mission_spec import MissionSpec
from design_agent.optimizer import OptimizationResult
from design_agent.parameterization import initial_guess
from design_agent.report import render_report, write_report
from design_agent.uncertainty import run_monte_carlo


def _fake_agent_report():
    """A complete AgentReport without running the optimizer (fast)."""
    spec = MissionSpec.example_trainer()
    d = initial_guess(spec, frozen_airfoil="naca2412")
    res = analyze(d, spec)
    records = evaluate_numeric(res, Margins().with_overrides(spec))
    optres = OptimizationResult(
        success=all(r.passed for r in records),
        design=d,
        mass_total_kg=scalar(res.mass.total),
        records=records,
        solver_stats={"alphas": {"cruise": 2.0, "loiter": 2.0, "sea": 8.0}},
        message="test fixture",
    )
    steps = [
        AgentStep("parse", "mission parsed: role='trainer'", 0.01),
        AgentStep("optimize", "fixture optimize step", 1.0),
        AgentStep("diagnose", "fixture diagnosis", 0.0),
    ]
    return AgentReport(spec=spec, result=optres, steps=steps,
                       advice=diagnose(optres))


def test_render_report_sections():
    rep = _fake_agent_report()
    md = render_report(rep)
    assert md.startswith("# Autonomous Aircraft Design Report")
    assert "## Mission specification" in md
    assert "## Agent audit trail" in md
    assert "## Design geometry" in md
    assert "## Mass breakdown" in md
    assert "## Constraint verification" in md
    assert "trim_cruise" in md
    assert "trainer" in md
    # every constraint record appears in the table
    for r in rep.result.records:
        assert r.name in md


def test_render_report_with_robustness_and_figures():
    rep = _fake_agent_report()
    rob = run_monte_carlo(rep.result.design, rep.spec, n_samples=2, seed=0,
                          trim_iterations=1)
    md = render_report(rep, robustness=rob,
                       figures={"Geometry": "figs/geometry.png"})
    assert "## Monte-Carlo robustness" in md
    assert "p(all constraints satisfied)" in md
    assert "![Geometry](figs/geometry.png)" in md


def test_write_report_creates_file(tmp_path):
    rep = _fake_agent_report()
    out = write_report(rep, tmp_path / "report.md")
    assert out.exists() and out.stat().st_size > 1_000
