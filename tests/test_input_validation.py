"""Offline tests for input validation and sanitisation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.investigator import MAX_QUESTION_LENGTH, investigate
from tests.test_investigator import FakeClient, FakeResponse, step


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"channel": ["Online", "Retail"], "revenue": [10.0, 20.0]})


def test_empty_question_is_rejected():
    report = investigate("", make_df(), client=FakeClient())
    assert report.status == "error"
    assert "non-empty" in (report.error or "").lower()


def test_oversized_question_is_rejected():
    long_question = "x" * (MAX_QUESTION_LENGTH + 1)
    report = investigate(long_question, make_df(), client=FakeClient())
    assert report.status == "error"
    assert "too long" in (report.error or "").lower()


def test_question_at_limit_is_accepted():
    question = "x" * MAX_QUESTION_LENGTH
    client = FakeClient(
        {"status": "unsupported", "reasoning": "no tool fits", "unsupported_reason": "no tool", "steps": []}
    )
    report = investigate(question, make_df(), client=client, interpret=False)
    assert report.status == "unsupported"


# (The app module is exercised end-to-end by tests/test_app.py via AppTest.)
