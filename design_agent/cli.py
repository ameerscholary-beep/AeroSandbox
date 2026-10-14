"""
Command-line interface for the design agent.

Usage:
    python -m design_agent.cli "Design a 4-seat piston trainer ..."
    python -m design_agent.cli --spec-json mission.json --out manifest.json

Exit code 0 if all constraints pass, 1 otherwise (CI-friendly).
"""

from __future__ import annotations

import argparse
import json
import sys

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


def _manifest(report) -> dict:
    try:
        spec_dict = report.spec.model_dump()
    except AttributeError:
        spec_dict = report.spec.dict()
    return {
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

    if args.out:
        with open(args.out, "w") as f:
            json.dump(_manifest(report), f, indent=2, default=str)
        print(f"\nmanifest written to {args.out}")

    print("\nRESULT:", "SUCCESS" if report.success else "CONSTRAINTS UNRESOLVED")
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
