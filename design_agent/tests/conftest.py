"""Shared fixtures for the design_agent test suite. No network access anywhere."""

from __future__ import annotations

import json

import pytest

from design_agent.mission_spec import MissionSpec


@pytest.fixture
def trainer_spec() -> MissionSpec:
    return MissionSpec.example_trainer()


@pytest.fixture
def feeder_spec() -> MissionSpec:
    return MissionSpec.example_feeder_transport()


@pytest.fixture
def surveillance_spec() -> MissionSpec:
    return MissionSpec.example_surveillance()


class _FakeMessage:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


class FakeOpenAI:
    """Minimal stand-in for the OpenAI client, scripted per-test."""

    def __init__(self, responses: list[str | Exception]):
        self._responses = list(responses)
        self.chat = type("chat", (), {})()
        self.chat.completions = self  # create() called directly on completions
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("FakeOpenAI: no scripted response left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


def fake_client_returning(spec: MissionSpec, *extra: str | Exception) -> FakeOpenAI:
    """Fake client whose first reply is the spec as JSON."""
    payload = json.dumps(spec.model_dump())
    return FakeOpenAI([payload, *extra])
