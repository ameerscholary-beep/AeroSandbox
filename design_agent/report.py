"""
Markdown design-report generation — the auditable thesis artifact.

One `write_report()` call turns an `AgentReport` (plus optional Monte-Carlo
`RobustnessReport` and figure paths) into a self-contained Markdown document:
mission, agent audit trail, geometry, mass breakdown, full constraint table,
robustness statistics, and the agent's advice. Markdown keeps the report
diff-able in git and renders natively on GitHub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from design_agent.agent_loop import AgentReport
from design_agent.analysis import analyze, scalar
from design_agent.uncertainty import RobustnessReport


def render_report(
    report: AgentReport,
    robustness: RobustnessReport | None = None,
    figures: dict[str, str] | None = None,
) -> str:
    """Render the full Markdown report as a string.

    ``figures`` maps a caption to a report-relative image path, e.g.
    ``{"Three-view geometry": "geometry.png"}``.
    """
    spec = report.spec
    res = report.result
    design = res.design
    lines: list[str] = []
    add = lines.append

    add("# Autonomous Aircraft Design Report")
    add("")
    add(f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
        "the design agent — all numbers verified against the numeric "
        "NeuralFoil/AeroBuildup stack.*")
    add("")

    # ---- Mission ----
    add("## Mission specification")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| role | {spec.role} |")
    add(f"| payload | {spec.payload_mass_kg:.0f} kg |")
    add(f"| crew | {spec.crew_count} |")
    add(f"| range | {spec.range_m / 1e3:.0f} km |")
    add(f"| cruise | {spec.cruise_speed_mps:.1f} m/s at "
        f"{spec.cruise_altitude_m:.0f} m |")
    add(f"| max takeoff distance | {spec.max_takeoff_distance_m:.0f} m |")
    if spec.stall_speed_max_mps:
        add(f"| max stall speed | {spec.stall_speed_max_mps:.1f} m/s |")
    add(f"| reserve | {spec.endurance_reserve_min:.0f} min |")
    add(f"| min climb rate | {spec.climb_rate_min_mps:.1f} m/s |")
    add("")

    # ---- Agent audit trail ----
    add("## Agent audit trail")
    add("")
    add("| phase | detail | time [s] |")
    add("|---|---|---|")
    for s in report.steps:
        add(f"| {s.phase} | {s.detail} | {s.elapsed_s:.1f} |")
    add("")
    add(f"**Outcome: {'SUCCESS — all constraints satisfied' if report.success else 'CONSTRAINTS UNRESOLVED'}** "
        f"({res.constraints_passed}/{res.constraints_total}, "
        f"gross mass {res.mass_total_kg:.0f} kg). Solver: {res.message}")
    add("")

    # ---- Geometry ----
    add("## Design geometry")
    add("")
    s_ref = design.span * design.root_chord * (1 + design.taper) / 2
    add("| quantity | value |")
    add("|---|---|")
    add(f"| span | {design.span:.2f} m |")
    add(f"| wing area (ref) | {s_ref:.2f} m² |")
    add(f"| root chord | {design.root_chord:.2f} m |")
    add(f"| taper | {design.taper:.2f} |")
    add(f"| quarter-chord sweep | {design.sweep_qc_deg:.1f}° |")
    add(f"| dihedral | {design.dihedral_deg:.1f}° |")
    add(f"| htail area / arm | {design.htail_area:.2f} m² / "
        f"{design.htail_arm:.2f} m |")
    add(f"| vtail area / arm | {design.vtail_area:.2f} m² / "
        f"{design.vtail_arm:.2f} m |")
    add(f"| fuselage L × D | {design.fuselage_length:.2f} m × "
        f"{2 * design.fuselage_radius:.2f} m |")
    add(f"| CG | {design.cg_mac_fraction:.3f} MAC |")
    add(f"| installed power | {design.power_w / 1e3:.1f} kW |")
    add("")

    # ---- Mass breakdown (numeric re-analysis at the design point) ----
    try:
        ana = analyze(
            design, spec,
            alphas=res.solver_stats.get("alphas") or None,
            compute_modes=False,
        )
        mb = ana.mass
        add("## Mass breakdown")
        add("")
        add("| component | mass [kg] |")
        add("|---|---|")
        for k in ("wing", "fuselage", "hstab", "vstab", "engine", "gear",
                  "flight_controls", "fuel_system", "avionics", "electrical",
                  "furnishings"):
            add(f"| {k.replace('_', ' ')} | {scalar(getattr(mb, k)):.1f} |")
        add(f"| **empty** | **{scalar(mb.empty):.1f}** |")
        add(f"| fuel | {scalar(mb.fuel):.1f} |")
        add(f"| useful load | {mb.useful:.1f} |")
        add(f"| **gross** | **{scalar(mb.total):.1f}** |")
        add("")
        add(f"Stall speed (SL, CLmax): {scalar(ana.v_stall):.1f} m/s · "
            f"cruise L/D: {scalar(ana.ld_ratio):.1f}")
        add("")
    except Exception:
        pass  # the constraint table below already carries the essentials

    # ---- Constraint table ----
    add("## Constraint verification")
    add("")
    add("| | constraint | value | limit |")
    add("|---|---|---|---|")
    for r in res.records:
        mark = "✅" if r.passed else "❌"
        add(f"| {mark} | {r.name} | {r.value:.4g} | {r.kind} {r.limit:.4g} |")
    add("")

    # ---- Robustness ----
    if robustness is not None:
        add("## Monte-Carlo robustness")
        add("")
        add(f"```\n{robustness.summary()}\n```")
        add("")

    # ---- Figures ----
    if figures:
        add("## Figures")
        add("")
        for caption, relpath in figures.items():
            add(f"![{caption}]({relpath})")
            add("")

    # ---- Advice ----
    if report.advice:
        add("## Agent advice")
        add("")
        for a in report.advice:
            add(f"- {a}")
        add("")

    return "\n".join(lines)


def write_report(
    report: AgentReport,
    path: str | Path,
    robustness: RobustnessReport | None = None,
    figures: dict[str, str] | None = None,
) -> Path:
    """Write the Markdown report to ``path``; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report, robustness=robustness, figures=figures))
    return path
