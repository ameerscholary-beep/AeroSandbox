"""Figure generation: geometry three-view, constraint dashboard, robustness."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from design_agent.analysis import analyze
from design_agent.constraints import Margins, evaluate_numeric
from design_agent.mission_spec import MissionSpec
from design_agent.parameterization import initial_guess
from design_agent.plots import plot_constraints, plot_geometry, plot_robustness
from design_agent.uncertainty import run_monte_carlo


def _trainer():
    spec = MissionSpec.example_trainer()
    return spec, initial_guess(spec, frozen_airfoil="naca2412")


def test_plot_geometry_saves(tmp_path):
    _spec, d = _trainer()
    out = tmp_path / "geometry.png"
    fig = plot_geometry(d, path=out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


def test_plot_constraints_saves(tmp_path):
    spec, d = _trainer()
    res = analyze(d, spec)
    records = evaluate_numeric(res, Margins().with_overrides(spec))
    out = tmp_path / "constraints.png"
    fig = plot_constraints(records, path=out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


def test_plot_robustness_saves(tmp_path):
    spec, d = _trainer()
    rep = run_monte_carlo(d, spec, n_samples=3, seed=2, trim_iterations=1)
    out = tmp_path / "robustness.png"
    fig = plot_robustness(rep, path=out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)
