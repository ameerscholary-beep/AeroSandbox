"""
Natural-language mission parser.

Primary path: an OpenAI-compatible chat model (JSON mode) producing a `MissionSpec`.
Safety net: one validation-feedback retry, then a deterministic rule-based extractor,
so a malformed LLM response never hard-fails the pipeline.

Every exchange is recorded in a `ParseTrace` for the reproducibility manifest.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from design_agent.mission_spec import MissionSpec

_PROMPT_PATH = Path(__file__).parent / "prompts" / "mission_parse.md"

DEFAULT_MODEL = "gpt-4o-mini"


class ParseError(ValueError):
    """Raised when neither the LLM nor the rule-based parser can build a spec."""


@dataclass
class ParseTrace:
    """Audit record of one parse attempt, serialized into the report manifest."""

    prompt: str = ""
    raw_response: str | None = None
    model: str | None = None
    attempts: int = 0
    used_fallback: bool = False
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text()


def _make_client() -> "Any":
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment


def parse_mission(
    text: str,
    client: Any | None = None,
    model: str | None = None,
) -> tuple[MissionSpec, ParseTrace]:
    """
    Parse natural-language mission text into a `MissionSpec`.

    Parameters
    ----------
    text : str
        The user's mission description.
    client : object | None
        An OpenAI-compatible client (for testing/mocking). If None, one is created,
        which requires ``OPENAI_API_KEY`` to be set.
    model : str | None
        Model name; defaults to ``AGENT_MODEL`` or `DEFAULT_MODEL`.

    Returns
    -------
    (MissionSpec, ParseTrace)
        The validated spec and a full audit trace.
    """
    model = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    trace = ParseTrace(model=model)
    template = _load_prompt_template()

    if client is None:
        try:
            client = _make_client()
        except Exception as exc:  # no credentials -> straight to fallback
            trace.errors.append(f"client_init: {exc}")

    if client is not None:
        for attempt in range(2):
            trace.attempts += 1
            prompt = (
                template.replace("{{mission_text}}", text)
                if attempt == 0
                else template.replace("{{mission_text}}", text)
                + "\n\nYour previous response failed validation:\n"
                + "\n".join(trace.errors)
                + "\nReturn a corrected JSON object only."
            )
            trace.prompt = prompt
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content or ""
                trace.raw_response = raw
                spec = MissionSpec.model_validate(json.loads(raw))
                return spec, trace
            except Exception as exc:
                trace.errors.append(f"attempt_{attempt}: {exc}")

    # Fallback: deterministic rule-based extraction
    trace.used_fallback = True
    try:
        return _RuleBasedParser().parse(text), trace
    except Exception as exc:
        trace.errors.append(f"fallback: {exc}")
        raise ParseError(
            f"Could not parse mission from text. Errors: {trace.errors}"
        ) from exc


# ---------------------------------------------------------------------------
# Rule-based fallback parser
# ---------------------------------------------------------------------------

_M_PER_UNIT = {
    "m": 1.0,
    "km": 1e3,
    "nm": 1852.0,
    "nmi": 1852.0,
    "mile": 1609.344,
    "miles": 1609.344,
    "mi": 1609.344,
    "ft": 0.3048,
}

_MPS_PER_UNIT = {
    "m/s": 1.0,
    "mps": 1.0,
    "km/h": 1.0 / 3.6,
    "kph": 1.0 / 3.6,
    "knot": 0.514444,
    "knots": 0.514444,
    "kt": 0.514444,
    "kts": 0.514444,
}


class _RuleBasedParser:
    """Keyword/regex extractor used when the LLM path is unavailable or fails."""

    def parse(self, text: str) -> MissionSpec:
        t = text.lower()

        seats = self._count(r"(\d+)\s*[- ]?\s*(?:seat|seater|passenger|pax)", t)
        pax = self._count(r"(\d+)\s+(?:passengers|pax)\b", t)
        occupants = max(seats or 0, pax or 0)
        unmanned = any(k in t for k in ("unmanned", "uav", "drone"))
        crew = 0 if unmanned else (1 if occupants >= 1 else 0)

        payload_kg = None
        m = re.search(r"payload[^0-9]*?(\d+)\s*kg", t)
        if m:
            payload_kg = float(m.group(1))
        elif occupants:
            payload_kg = max(occupants - crew, 0) * 90.0
        else:
            raise ParseError("No payload or passenger count found in text.")

        range_m = self._quantity_with_unit(
            r"(\d+(?:\.\d+)?)\s*(km|nm|nmi|miles|mile|mi|ft)\s*(?:range|ferry)", t,
            _M_PER_UNIT,
        )
        if range_m is None:
            range_m = self._quantity_with_unit(
                r"(?:range|ferry)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(km|nm|nmi|miles|mile|mi|ft)",
                t,
                _M_PER_UNIT,
            )
        if range_m is None:
            raise ParseError("No range found in text.")

        cruise_speed = self._near(
            t, r"\bcruise\b", r"(\d+(?:\.\d+)?)\s*(m/s|mps|km/h|kph|knots|knot|kt|kts)\b",
            _MPS_PER_UNIT,
        ) or self._quantity_with_unit(
            r"(\d+(?:\.\d+)?)\s*(km/h|knots|kt)\b", t, _MPS_PER_UNIT
        )
        if cruise_speed is None:
            raise ParseError("No cruise speed found in text.")

        altitude = self._near(
            t, r"\bcruise\b|\bat\b|\baltitude\b|\balt\b",
            r"(\d+(?:\.\d+)?)\s*(km|ft|m)\b(?!\s*/)", _M_PER_UNIT,
        )
        altitude = 0.0 if altitude is None else altitude

        tofl = self._near(
            t, r"\btakeoff\b|\btake-off\b|\btofl\b|\brunway\b|\bfield\b",
            r"(\d+(?:\.\d+)?)\s*(m|ft)\b", _M_PER_UNIT,
        )
        stall = self._near(
            t, r"\bstall\b",
            r"(\d+(?:\.\d+)?)\s*(m/s|km/h|knots|kt)\b", _MPS_PER_UNIT,
        )
        reserve = self._minutes(t, r"(\d+)\s*min(?:ute)?s?[^.]*?(?:reserve|loiter)") or (
            self._minutes(t, r"(?:reserve|loiter)[^.]*?(\d+)\s*min")
        )

        role_m = re.match(r"(?:design|create|build)?\s*(?:a|an)?\s*(.+?)(?:[,.]|$)", text.strip())
        role = role_m.group(1).strip() if role_m else "aircraft"

        return MissionSpec(
            role=role,
            payload_mass_kg=payload_kg,
            crew_count=crew,
            range_m=range_m,
            cruise_speed_mps=cruise_speed,
            cruise_altitude_m=min(altitude, 20_000.0),
            max_takeoff_distance_m=tofl,
            stall_speed_max_mps=stall,
            endurance_reserve_min=reserve if reserve is not None else 45.0,
            climb_rate_min_mps=5.0,
        )

    @staticmethod
    def _count(pattern: str, t: str) -> int | None:
        m = re.search(pattern, t)
        return int(m.group(1)) if m else None

    @staticmethod
    def _quantity_with_unit(pattern: str, t: str, table: dict[str, float]) -> float | None:
        m = re.search(pattern, t)
        if not m:
            return None
        return float(m.group(1)) * table[m.group(2)]

    def _near(self, t: str, keywords: str, pattern: str, table: dict[str, float]) -> float | None:
        """Find the quantity closest to a keyword in its sentence.

        Prefers matches *after* the keyword ("stall under 30 m/s"); falls back to
        the nearest match before it ("a 1000 m runway").
        """
        best_key: tuple[int, float] | None = None
        best_val: float | None = None
        for sentence in re.split(r"[.;\n]", t):
            kw = re.search(keywords, sentence)
            if not kw:
                continue
            kw_c = (kw.start() + kw.end()) / 2
            for m in re.finditer(pattern, sentence):
                m_c = (m.start() + m.end()) / 2
                d = m_c - kw_c
                key = (0 if d >= 0 else 1, abs(d))  # after-keyword first, then nearest
                if best_key is None or key < best_key:
                    best_key = key
                    best_val = float(m.group(1)) * table[m.group(2)]
        return best_val

    @staticmethod
    def _minutes(t: str, pattern: str) -> float | None:
        m = re.search(pattern, t)
        return float(m.group(1)) if m else None
