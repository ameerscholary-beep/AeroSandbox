"""Monte-Carlo robustness: Dispersion wiring, report statistics, determinism."""

import pytest

from design_agent.analysis import Dispersion, analyze, scalar
from design_agent.mission_spec import MissionSpec
from design_agent.parameterization import initial_guess
from design_agent.uncertainty import (
    RobustnessReport,
    UncertaintyModel,
    run_monte_carlo,
)


def _trainer():
    spec = MissionSpec.example_trainer()
    return spec, initial_guess(spec, frozen_airfoil="naca2412")


def test_dispersion_mass_factor_grows_mass():
    spec, d = _trainer()
    base = analyze(d, spec)
    heavy = analyze(d, spec, dispersion=Dispersion(mass_factor=1.2))
    # Empty mass grows by *more* than 1.2x: the numeric TOGW closure feeds the
    # higher empty mass back into the TOGW-sensitive components (wing, gear).
    assert scalar(heavy.mass.empty) > 1.1 * scalar(base.mass.empty)
    assert scalar(heavy.mass.total) > 1.05 * scalar(base.mass.total)


def test_dispersion_density_scales_forces_first_order():
    spec, d = _trainer()
    base = analyze(d, spec)
    thin = analyze(d, spec, dispersion=Dispersion(density_factor=0.8))
    assert scalar(thin.aero["L"][0]) == pytest.approx(
        0.8 * scalar(base.aero["L"][0]), rel=1e-6
    )
    assert scalar(thin.aero["D"][0]) == pytest.approx(
        0.8 * scalar(base.aero["D"][0]), rel=1e-6
    )
    # thinner air at fixed CLmax and similar weight -> higher stall speed
    assert scalar(thin.v_stall) > scalar(base.v_stall)


def test_dispersion_temperature_raises_stall_speed():
    spec, d = _trainer()
    base = analyze(d, spec)
    hot = analyze(d, spec, dispersion=Dispersion(temperature_offset_k=20.0))
    assert scalar(hot.v_stall) > scalar(base.v_stall)


def test_monte_carlo_report_statistics():
    spec, d = _trainer()
    rep = run_monte_carlo(d, spec, n_samples=5, seed=1, trim_iterations=1)
    assert isinstance(rep, RobustnessReport)
    assert rep.n_samples == 6  # 5 dispersed + 1 nominal
    assert 0.0 <= rep.p_success <= 1.0

    pp = rep.pass_probability()
    assert len(pp) >= 12  # full constraint catalog was verified per sample
    assert all(0.0 <= p <= 1.0 for p in pp.values())

    mp = rep.mass_percentiles()
    assert mp["p5"] <= mp["p50"] <= mp["p95"]

    text = rep.summary()
    assert "p(all constraints satisfied)" in text
    assert "pass probabilities" in text


def test_monte_carlo_retrims_samples():
    """Samples must be re-trimmed: cruise alpha moves away from the seed."""
    spec, d = _trainer()
    rep = run_monte_carlo(
        d, spec, n_samples=3, seed=3, trim_iterations=2,
        model=UncertaintyModel(mass_factor_sigma=0.15),  # force big weight shifts
    )
    moved = [
        s for s in rep.samples[1:]  # skip nominal
        if s.alphas and abs(s.alphas["cruise"] - 2.0) > 0.05
    ]
    assert moved, "no sample was re-trimmed away from the baseline alpha"


def test_monte_carlo_deterministic_for_fixed_seed():
    spec, d = _trainer()
    r1 = run_monte_carlo(d, spec, n_samples=3, seed=7, trim_iterations=1)
    r2 = run_monte_carlo(d, spec, n_samples=3, seed=7, trim_iterations=1)
    m1 = [s.mass_total_kg for s in r1.samples]
    m2 = [s.mass_total_kg for s in r2.samples]
    assert m1 == pytest.approx(m2, rel=0.0, abs=1e-9)
    assert [s.success for s in r1.samples] == [s.success for s in r2.samples]
