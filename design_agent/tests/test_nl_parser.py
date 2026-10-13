import json

import pytest

from design_agent.mission_spec import MissionSpec
from design_agent.nl_parser import ParseError, parse_mission
from design_agent.tests.conftest import FakeOpenAI, fake_client_returning


def test_parse_happy_path(trainer_spec):
    client = fake_client_returning(trainer_spec)
    spec, trace = parse_mission("any text", client=client, model="fake")
    assert spec == trainer_spec
    assert trace.attempts == 1
    assert not trace.used_fallback
    assert trace.model == "fake"
    assert trace.raw_response  # audit trail captured


def test_parse_retries_on_invalid_json_then_succeeds(trainer_spec):
    client = fake_client_returning(trainer_spec)
    client._responses.insert(0, "this is not json")
    spec, trace = parse_mission("any text", client=client)
    assert spec == trainer_spec
    assert trace.attempts == 2
    assert len(trace.errors) == 1


def test_parse_falls_back_to_rule_parser():
    client = FakeOpenAI([RuntimeError("boom"), RuntimeError("boom")])
    text = (
        "Design a 4-seat trainer, 800 km range, cruise 60 m/s at 3 km altitude, "
        "takeoff under 600 m, stall under 30 m/s."
    )
    spec, trace = parse_mission(text, client=client)
    assert trace.used_fallback
    assert spec.payload_mass_kg == pytest.approx(3 * 90.0)  # 4 seats - 1 crew
    assert spec.range_m == pytest.approx(800e3)
    assert spec.cruise_speed_mps == pytest.approx(60.0)
    assert spec.cruise_altitude_m == pytest.approx(3000.0)
    assert spec.max_takeoff_distance_m == pytest.approx(600.0)
    assert spec.stall_speed_max_mps == pytest.approx(30.0)
    assert spec.crew_count == 1


def test_parse_rule_parser_units():
    client = FakeOpenAI([RuntimeError("x"), RuntimeError("x")])
    text = "A 2-seat aircraft, range 200 nm, cruise 120 knots at 10000 ft."
    spec, trace = parse_mission(text, client=client)
    assert spec.range_m == pytest.approx(200 * 1852.0, rel=1e-3)
    assert spec.cruise_speed_mps == pytest.approx(120 * 0.514444, rel=1e-3)
    assert spec.cruise_altitude_m == pytest.approx(10000 * 0.3048, rel=1e-3)


def test_parse_raises_when_nothing_found():
    client = FakeOpenAI([RuntimeError("x"), RuntimeError("x")])
    with pytest.raises(ParseError):
        parse_mission("hello there, general kenobi", client=client)


def test_trace_serializable(trainer_spec):
    client = fake_client_returning(trainer_spec)
    _, trace = parse_mission("x", client=client)
    json.dumps(trace.__dict__)  # must not raise
