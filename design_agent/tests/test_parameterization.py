import numpy as np
import pytest

from design_agent.parameterization import (
    N_CST_WEIGHTS,
    DesignVector,
    initial_guess,
)


def _specs():
    from design_agent.mission_spec import MissionSpec

    return [
        MissionSpec.example_trainer(),
        MissionSpec.example_feeder_transport(),
        MissionSpec.example_surveillance(),
    ]


@pytest.mark.parametrize("frozen", ["naca2412", None])
def test_flat_roundtrip(frozen):
    x = initial_guess(_specs()[0], frozen_airfoil=frozen)
    x2 = DesignVector.from_flat(x.to_flat(), frozen_airfoil=frozen)
    assert np.allclose(x.to_flat(), x2.to_flat())


def test_field_specs_counts():
    assert len(DesignVector.field_specs("naca2412")) == 14
    cst = DesignVector.field_specs(None)
    assert len(cst) == 18  # + 4 airfoil blocks (2 vectors of 8 + 2 scalars)
    total = sum(s.size for s in cst)
    assert total == 14 + 16 + 2


@pytest.mark.parametrize("frozen", ["naca2412", None])
def test_initial_guess_within_bounds(frozen):
    for spec in _specs():
        x = initial_guess(spec, frozen_airfoil=frozen)
        flat = x.to_flat()
        for s in DesignVector.field_specs(frozen):
            block = flat
            lo, hi = s.lo, s.hi
            assert np.all(block >= -1e9)  # sanity
        assert x.span >= 6.0 and x.span <= 22.0
        assert x.taper >= 0.2 and x.taper <= 1.0
        assert 20e3 <= x.power_w <= 800e3
        if frozen is None:
            assert x.upper_weights.shape == (N_CST_WEIGHTS,)
            assert np.all(x.upper_weights <= 0.55)


def test_clamp():
    spec = _specs()[0]
    x = initial_guess(spec, frozen_airfoil="naca2412")
    x = DesignVector(**{**x.__dict__, "span": 99.0, "power_w": 1.0})
    xc = x.clamp()
    assert xc.span == 22.0
    assert xc.power_w == pytest.approx(20e3)


def test_to_airplane_geometry():
    spec = _specs()[0]
    x = initial_guess(spec, frozen_airfoil="naca2412")
    ap = x.to_airplane()
    S, mac, x_le_mac = x.wing_aero_numbers()
    assert ap.s_ref == pytest.approx(S)
    assert ap.c_ref == pytest.approx(mac)
    assert ap.b_ref == pytest.approx(x.span)
    assert len(ap.wings) == 3
    assert len(ap.fuselages) == 1
    # analytic trapezoid area check
    S_hand = x.span / 2 * x.root_chord * (1 + x.taper)
    assert S == pytest.approx(S_hand)


def test_perturb_is_seeded():
    spec = _specs()[0]
    x = initial_guess(spec, frozen_airfoil="naca2412")
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    p1 = x.perturb(rng1)
    p2 = x.perturb(rng2)
    assert np.allclose(p1.to_flat(), p2.to_flat())
    assert not np.allclose(p1.to_flat(), x.to_flat())


def test_cst_mode_builds_airplane():
    spec = _specs()[0]
    x = initial_guess(spec, frozen_airfoil=None)
    ap = x.to_airplane()
    assert ap.wings[0].xsecs[0].airfoil.name == "agent_cst"
