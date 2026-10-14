from dataclasses import replace

from design_agent.analysis import analyze
from design_agent.constraints import (
    Margins,
    all_passed,
    evaluate_numeric,
    takeoff_ground_roll,
    violations,
)
from design_agent.parameterization import initial_guess


def _numeric(trainer_spec, frozen="naca2412"):
    x = initial_guess(trainer_spec, frozen_airfoil=frozen)
    return analyze(x, trainer_spec)


def test_margins_floors():
    m = Margins(static_min=0.0, thrust=1.0, cl_operating=1.5, tofl_usage=1.5,
                zeta_short_period=0.0).clamped()
    assert m.static_min == 0.03
    assert m.thrust == 1.05
    assert m.cl_operating == 0.95
    assert m.tofl_usage == 0.85
    assert m.zeta_short_period == 0.30 * 0.7


def test_margins_overrides_from_spec(trainer_spec):
    spec = trainer_spec.model_copy(update={"constraints_overrides": {"thrust": 1.35}})
    m = Margins().with_overrides(spec)
    assert m.thrust == 1.35
    # unknown override keys ignored
    spec2 = trainer_spec.model_copy(update={"constraints_overrides": {"nonsense": 1.0}})
    assert Margins().with_overrides(spec2) == Margins().clamped()


def test_evaluate_numeric_structure(trainer_spec):
    res = _numeric(trainer_spec)
    records = evaluate_numeric(res, Margins())
    names = {r.name for r in records}
    assert {"trim_cruise", "cl_margin", "thrust_margin", "static_margin",
            "zeta_short_period", "takeoff_roll", "nf_confidence"} <= names
    assert all(isinstance(r.passed, bool) for r in records)
    # trim should hold for the default alpha guesses? (alpha is a guess, not trim-solved)
    trim = next(r for r in records if r.name == "trim_cruise")
    assert abs(trim.value - trim.limit) / trim.limit < 0.5  # within 50%: un-solved guess


def test_evaluate_detects_violation(trainer_spec):
    res = _numeric(trainer_spec)
    strict = replace(Margins().clamped(), thrust=99.0, static_min=0.9)
    records = evaluate_numeric(res, strict)
    failed = {r.name for r in violations(records)}
    assert "thrust_margin" in failed
    assert "static_margin_max" in failed  # init CG/tail gives a large static margin
    assert not all_passed(records)


def test_takeoff_roll_positive_and_library_check(trainer_spec):
    res = _numeric(trainer_spec)
    roll = takeoff_ground_roll(res, Margins())
    assert float(roll) > 0
    records = evaluate_numeric(res, Margins())
    lib = next((r for r in records if r.name == "takeoff_roll_library"), None)
    assert lib is not None  # trainer spec sets a TOFL limit
    assert lib.value > 0


def test_margins_respected_between_surrogate_and_library(trainer_spec):
    # surrogate uses 80% of limit; library check uses 100%: both should be present
    res = _numeric(trainer_spec)
    records = evaluate_numeric(res, Margins())
    s = next(r for r in records if r.name == "takeoff_roll")
    l = next(r for r in records if r.name == "takeoff_roll_library")
    assert s.limit < l.limit  # surrogate budget tighter than raw limit
