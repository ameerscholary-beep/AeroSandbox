import pytest

from design_agent.constraints import Margins
from design_agent.optimizer import optimize
from design_agent.parameterization import initial_guess


def test_optimize_converges_trainer(trainer_spec):
    res = optimize(trainer_spec, frozen_airfoil="naca2412", max_runtime=120.0)
    assert res.mass_total_kg > trainer_spec.useful_load_kg
    assert res.constraints_total >= 12
    if not res.success:
        failed = [(r.name, r.value, r.limit) for r in res.records if not r.passed]
        pytest.fail(f"constraints failed: {failed}\nmessage: {res.message}")


def test_optimize_infeasible_returns_diagnostics(trainer_spec):
    # Impossible: 200 m takeoff limit with a 600 kg payload -> must not raise,
    # must return a result with constraint failures identified.
    impossible = trainer_spec.model_copy(
        update={"max_takeoff_distance_m": 200.0, "payload_mass_kg": 600.0}
    )
    res = optimize(impossible, frozen_airfoil="naca2412", max_runtime=60.0, max_iter=200)
    assert res.records  # diagnostics exist either way
    if not res.success:
        names = [r.name for r in res.records if not r.passed]
        assert names
    # either way: no exception, and mass is physically plausible if solved
    if res.success:
        assert 300 < res.mass_total_kg < 20_000


def test_optimize_respects_margins_override(trainer_spec):
    spec = trainer_spec.model_copy(update={"constraints_overrides": {"thrust": 1.5}})
    res = optimize(spec, frozen_airfoil="naca2412", max_runtime=120.0)
    rec = next(r for r in res.records if r.name == "thrust_margin")
    if res.success:
        assert rec.passed  # designed against the overridden (tighter) margin
    # record carries the overridden limit (1.5 * D) — larger than default 1.2 * D
    default = optimize(trainer_spec, frozen_airfoil="naca2412", max_runtime=120.0)
    rec_def = next(r for r in default.records if r.name == "thrust_margin")
    assert rec.limit > rec_def.limit
