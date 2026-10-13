import pytest
from pydantic import ValidationError

from design_agent.mission_spec import MissionSpec


def test_example_specs_are_valid():
    for spec in (
        MissionSpec.example_trainer(),
        MissionSpec.example_feeder_transport(),
        MissionSpec.example_surveillance(),
    ):
        assert spec.range_m > 0
        assert spec.cruise_speed_mps > 0
        assert spec.useful_load_kg > 0


def test_useful_load():
    spec = MissionSpec.example_trainer()  # 340 kg payload + 1 x 90 kg crew
    assert spec.useful_load_kg == pytest.approx(430.0)


def test_negative_payload_rejected():
    with pytest.raises(ValidationError):
        MissionSpec(
            payload_mass_kg=-10, range_m=1e5, cruise_speed_mps=50, cruise_altitude_m=1000
        )


def test_stall_above_cruise_rejected():
    with pytest.raises(ValidationError):
        MissionSpec(
            payload_mass_kg=100,
            range_m=1e5,
            cruise_speed_mps=50,
            cruise_altitude_m=1000,
            stall_speed_max_mps=60,
        )


def test_json_roundtrip(tmp_path):
    spec = MissionSpec.example_feeder_transport()
    p = tmp_path / "spec.json"
    spec.save_json(p)
    loaded = MissionSpec.load_json(p)
    assert loaded == spec


def test_prompt_dict_contains_converted_units():
    d = MissionSpec.example_trainer().to_prompt_dict()
    assert d["range_km"] == pytest.approx(800.0)
    assert d["cruise_speed_m_per_s"] == pytest.approx(60.0)
