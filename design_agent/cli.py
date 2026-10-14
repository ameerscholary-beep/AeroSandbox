"""
Command-line interface for the design agent.

Usage:
    python -m design_agent.cli "Design a 4-seat piston trainer ..."
    python -m design_agent.cli --spec-json mission.json --out manifest.json
    python -m design_agent.cli "..." --robustness 32 --figures figs/ --report report.md

Exit code 0 if all constraints pass, 1 otherwise (CI-friendly).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from design_agent.agent_loop import AgentConfig, DesignAgent
from design_agent.mission_spec import MissionSpec


def _load_spec_json(path: str) -> MissionSpec:
    text = open(path).read()
    try:
        return MissionSpec.model_validate_json(text)  # pydantic v2
    except AttributeError:
        return MissionSpec.parse_raw(text)  # pydantic v1 fallback


def _print_records(report) -> None:
    print("\nconstraint check:")
    for r in report.result.records:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {status}  {r.name:22s} value={r.value:12.5g}  "
            f"limit={r.limit:12.5g}  ({r.description})"
        )


def _manifest(report, robustness=None) -> dict:
    try:
        spec_dict = report.spec.model_dump()
    except AttributeError:
        spec_dict = report.spec.dict()
    out = {
        "mission": spec_dict,
        "success": report.success,
        "mass_total_kg": report.result.mass_total_kg,
        "constraints_passed": report.result.constraints_passed,
        "constraints_total": report.result.constraints_total,
        "message": report.result.message,
        "solver_stats": report.result.solver_stats,
        "records": [
            {
                "name": r.name,
                "value": r.value,
                "limit": r.limit,
                "kind": r.kind,
                "passed": r.passed,
                "description": r.description,
            }
            for r in report.result.records
        ],
        "advice": report.advice,
        "steps": [
            {"phase": s.phase, "detail": s.detail, "elapsed_s": s.elapsed_s}
            for s in report.steps
        ],
    }
    if robustness is not None:
        out["robustness"] = {
            "n_samples": robustness.n_samples,
            "p_success": robustness.p_success,
            "mass_percentiles": robustness.mass_percentiles(),
            "pass_probability": robustness.pass_probability(),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="design_agent",
        description="Natural-language mission -> optimized, verified aircraft design.",
    )
    ap.add_argument("mission", nargs="?", default=None,
                    help="natural-language mission statement")
    ap.add_argument("--spec-json", default=None,
                    help="path to a MissionSpec JSON file (skips NL parsing)")
    ap.add_argument("--airfoil", default="naca2412",
                    help="frozen database airfoil (default: naca2412)")
    ap.add_argument("--max-runtime", type=float, default=180.0,
                    help="per-solve IPOPT time budget [s]")
    ap.add_argument("--restarts", type=int, default=2,
                    help="perturbed-seed retries if constraints fail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write a JSON manifest here")
    ap.add_argument("--robustness", type=int, default=0, metavar="N",
                    help="run N Monte-Carlo robustness samples after design")
    ap.add_argument("--figures", default=None, metavar="DIR",
                    help="write geometry.png / constraints.png / robustness.png")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="write a Markdown design report here")
    args = ap.parse_args(argv)

    if not args.mission and not args.spec_json:
        ap.error("give a mission statement or --spec-json")

    agent = DesignAgent(AgentConfig(
        frozen_airfoil=args.airfoil,
        max_runtime=args.max_runtime,
        restarts=args.restarts,
        seed=args.seed,
    ))

    if args.spec_json:
        report = agent.design_from_spec(_load_spec_json(args.spec_json))
    else:
        report = agent.design_from_text(args.mission)

    print("\nagent audit trail:")
    for s in report.steps:
        print(f"  [{s.phase:8s}] {s.detail}  ({s.elapsed_s:.1f} s)")

    print(f"\ndesign gross mass: {report.result.mass_total_kg:.1f} kg")
    print(f"constraints: {report.result.constraints_passed}/"
          f"{report.result.constraints_total} passed")
    _print_records(report)

    if report.advice:
        print("\nagent advice:")
        for line in report.advice:
            print(f"  - {line}")

    # ---- Monte-Carlo robustness (optional) ----
    robustness = None
    if args.robustness > 0:
        from design_agent.uncertainty import run_monte_carlo

        robustness = run_monte_carlo(
            report.result.design,
            report.spec,
            alphas=report.result.solver_stats.get("alphas"),
            n_samples=args.robustness,
            seed=args.seed,
        )
        print("\n" + robustness.summary())

    # ---- Figures (optional) ----
    figure_paths: dict[str, str] = {}
    if args.figures:
        from design_agent.plots import (
            plot_constraints,
            plot_geometry,
            plot_robustness,
        )

        figdir = Path(args.figures)
        figdir.mkdir(parents=True, exist_ok=True)
        plot_geometry(report.result.design, path=figdir / "geometry.png")
        figure_paths["Three-view geometry"] = "geometry.png"
        plot_constraints(report.result.records, path=figdir / "constraints.png")
        figure_paths["Constraint verification"] = "constraints.png"
        if robustness is not None:
            plot_robustness(robustness, path=figdir / "robustness.png")
            figure_paths["Monte-Carlo robustness"] = "robustness.png"
        print(f"\nfigures written to {figdir}/")

    # ---- Report (optional) ----
    if args.report:
        from design_agent.report import write_report

        figs = None
        if figure_paths:
            # report-relative figure links: figures dir relative to report dir
            figdir_rel = Path(args.figures)
            try:
                rel = figdir_rel.relative_to(Path(args.report).parent)
            except ValueError:
                rel = figdir_rel
            figs = {cap: str(rel / p) for cap, p in figure_paths.items()}
        write_report(report, args.report, robustness=robustness, figures=figs)
        print(f"report written to {args.report}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(_manifest(report, robustness), f, indent=2, default=str)
        print(f"manifest written to {args.out}")

    print("\nRESULT:", "SUCCESS" if report.success else "CONSTRAINTS UNRESOLVED")
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
