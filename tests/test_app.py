"""Offline smoke tests for the Streamlit interface (AppTest, no browser)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.investigator as inv_mod

APP_PATH = str(PROJECT_ROOT / "app.py")


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)
        self.model = "fake-model"
        self.usage = {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 0,
            "totalTokenCount": 2,
        }

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, *responses):
        self._responses = list(responses) or [
            {
                "status": "plan",
                "reasoning": "count channels",
                "steps": [{"tool": "value_counts_table", "params": {"column": "channel"}, "purpose": "counts"}],
            },
            {
                "answer": "Online is the most common channel.",
                "interpretation": "The counts table shows Online leading.",
                "confidence": "high",
                "caveats": [],
            },
        ]
        self.calls = []

    def generate_json(self, prompt, *, schema=None, system_instruction=None, **kwargs):
        self.calls.append(prompt)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        import src.llm_client as lc

        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


@pytest.fixture
def patch_happy(monkeypatch):
    monkeypatch.setattr(inv_mod, "GeminiClient", lambda *a, **k: FakeClient())


@pytest.fixture
def patch_failing(monkeypatch):
    import src.llm_client as lc

    class FailingClient(FakeClient):
        def generate_json(self, prompt, *, schema=None, system_instruction=None, **kwargs):
            raise lc.LLMError("planning failed")

    monkeypatch.setattr(inv_mod, "GeminiClient", lambda *a, **k: FailingClient())


def _texts(at, attr):
    try:
        return [str(w.value) for w in getattr(at, attr)]
    except Exception:
        return []


def test_app_empty_state_renders(patch_happy):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    combined = "\n".join(_texts(at, "markdown") + _texts(at, "info"))
    assert "Upload a CSV" in combined or "Choose a CSV" in combined


def test_app_loads_sample(patch_happy):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    at.button(key="use_sample").click().run()
    assert not at.exception
    combined = "\n".join(_texts(at, "markdown") + _texts(at, "caption"))
    assert "sample_sales" in combined.lower() or "rows" in combined.lower()


def test_app_runs_investigation(patch_happy):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="use_sample").click().run()
    assert not at.exception
    at.text_input(key="question").set_value("Which channel is most common?").run()
    assert not at.exception
    at.button(key="investigate").click().run()
    assert not at.exception, at.exception
    combined = "\n".join(_texts(at, "markdown") + _texts(at, "success") + _texts(at, "info") + _texts(at, "warning"))
    assert "Findings (computed in Python)" in combined
    assert "Investigation complete" in combined


def test_app_surfaces_failure(patch_failing):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="use_sample").click().run()
    assert not at.exception
    at.text_input(key="question").set_value("Which channel is most common?").run()
    at.button(key="investigate").click().run()
    assert not at.exception, at.exception
    combined = "\n".join(_texts(at, "markdown") + _texts(at, "error"))
    assert "could not be completed" in combined.lower() or "planning failed" in combined.lower()
