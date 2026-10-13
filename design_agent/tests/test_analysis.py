import pytest

import aerosandbox as asb

from design_agent.analysis import analyze, is_symbolic, scalar
from design_agent.parameterization import initial_guess


def test_numeric_analysis(trainer_spec):
    x = initial_guess(trainer_spec, frozen_airfoil="naca2412")
    res = analyze(x, trainer_spec)
    assert res.is_numeric
    assert scalar(res.mass.total) > trainer_spec.useful_load_kg
    assert scalar(res.mass.empty) > 0
    assert scalar(res.mass.fuel) > 0
    # closure: total = useful + fuel + empty
    assert scalar(res.mass.total) == pytest.approx(
        res.mass.useful + scalar(res.mass.fuel) + scalar(res.mass.empty), rel=1e-6
    )
    # aero sanity
    assert scalar(res.aero["L"][0]) > 0
    assert scalar(res.ld_ratio) > 3
    assert 0.5 < scalar(res.cl_max_2d) < 2.2
    assert 0.0 < scalar(res.confidence) <= 1.0
    assert scalar(res.v_stall) > 10
    # modes computed numerically
    assert res.modes is not None
    for mode in ("phugoid", "short_period", "roll_subsidence", "dutch_roll", "spiral"):
        assert mode in res.modes


def test_numeric_fixed_point_converged(trainer_spec):
    x = initial_guess(trainer_spec, frozen_airfoil="naca2412")
    res = analyze(x, trainer_spec)
    # Raymer closure consistency: re-running with converged TOGW changes little
    m1 = scalar(res.mass.empty)
    res2 = analyze(x, trainer_spec)
    assert scalar(res2.mass.empty) == pytest.approx(m1, rel=1e-3)


def test_symbolic_analysis_traces(trainer_spec):
    opti = asb.Opti()
    import design_agent.parameterization as param

    x0 = initial_guess(trainer_spec, frozen_airfoil="naca2412")
    flat0 = x0.to_flat()
    specs = x0.field_specs("naca2412")
    flat_vars, offset = [], 0
    for s in specs:
        block = flat0[offset : offset + s.size]
        offset += s.size
        kw = dict(init_guess=block, scale=s.scale, log_transform=s.log_transform)
        v = opti.variable(lower_bound=s.lo, upper_bound=s.hi, **kw) if s.size == 1 else \
            opti.variable(init_guess=block, lower_bound=s.lo, upper_bound=s.hi, scale=s.scale)
        flat_vars.append(v)
    import aerosandbox.numpy as np

    xs = param.DesignVector.from_flat(np.concatenate(flat_vars), frozen_airfoil="naca2412")
    weight_n = opti.variable(init_guess=1500 * 9.81, lower_bound=500 * 9.81,
                             upper_bound=20_000 * 9.81, scale=15_000)
    alphas = {k: opti.variable(init_guess=2.0, lower_bound=-5, upper_bound=15)
              for k in ("cruise", "loiter", "sea")}
    res = analyze(xs, trainer_spec, alphas=alphas, weight_n=weight_n)
    assert is_symbolic(res.mass.total)
    assert is_symbolic(res.aero["CL"][0])
    assert is_symbolic(res.confidence)
    # trim + closure constraints must be registrable
    opti.subject_to(res.aero["L"][0] == weight_n)
    opti.subject_to(res.aero["L"][1] == weight_n * 1.0)
    opti.subject_to(weight_n == res.mass.total * 9.81)
    opti.minimize(res.mass.total)
    sol = opti.solve(verbose=False)
    assert 500 < sol(res.mass.total) < 20_000
