"""Offline tests for src.audit (safe logging)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.audit import (
    DEFAULT_LOG_PATH,
    build_log_record,
    log_investigation,
    read_log,
    token_summary,
)
from src.investigator import InvestigationReport, StepResult


def make_report(status: str = "ok", error: str | None = None) -> InvestigationReport:
    step = StepResult(
        tool="value_counts_table",
        params={"column": "channel"},
        result=__import__("pandas").DataFrame({"channel": ["Online"], "count": [94]}),
    )
    return InvestigationReport(
        question="Which channel is most common?",
        status=status,
        steps=[step],
        model="gemini-3.1-flash-lite",
        usage={"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 0, "totalTokenCount": 15},
        error=error,
        unverified_numbers=["999"],
    )


# ---------------------------------------------------------------------------
# token_summary
# ---------------------------------------------------------------------------
def test_token_summary_empty():
    assert token_summary(None) == {"prompt": 0, "output": 0, "thinking": 0, "total": 0}


def test_token_summary_values():
    usage = {"promptTokenCount": 1, "candidatesTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 6}
    assert token_summary(usage) == {"prompt": 1, "output": 2, "thinking": 3, "total": 6}


# ---------------------------------------------------------------------------
# build_log_record
# ---------------------------------------------------------------------------
def test_build_log_record_fields():
    report = make_report()
    fixed = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
    record = build_log_record(
        question=report.question,
        report=report,
        dataset_label="sample_sales.csv",
        n_rows=180,
        n_columns=12,
        redact_columns=["order_id"],
        latency_ms=1234,
        timestamp=fixed,
    )
    assert record["timestamp"] == "2026-11-16T12:00:00+00:00"
    assert record["question"] == "Which channel is most common?"
    assert record["dataset"] == {"label": "sample_sales.csv", "n_rows": 180, "n_columns": 12}
    assert record["redacted_columns"] == ["order_id"]
    assert record["status"] == "ok"
    assert record["model"] == "gemini-3.1-flash-lite"
    assert record["tools"] == ["value_counts_table"]
    assert record["parameters"] == {"value_counts_table": {"column": "channel"}}
    assert record["steps_ok"] == 1
    assert record["steps_failed"] == 0
    assert record["unverified_numbers"] == 1
    assert record["tokens"]["total"] == 15
    assert record["latency_ms"] == 1234
    assert record["error"] == ""


def test_build_log_record_clips_question():
    long_question = "x" * 500
    report = make_report()
    report.question = long_question
    record = build_log_record(question=long_question, report=report)
    assert len(record["question"]) <= 201
    assert record["question"].endswith("\u2026")


def test_build_log_record_clips_error():
    report = make_report(error="e" * 500)
    record = build_log_record(question="q", report=report)
    assert len(record["error"]) <= 201


def test_build_log_record_no_cell_values():
    """The record must never contain a computed cell value from the evidence."""
    report = make_report()
    record = build_log_record(question="q", report=report, n_rows=180, n_columns=12)
    serialised = json.dumps(record, default=str)
    # "Online" is a cell value in the result frame; it must not leak.
    assert "Online" not in serialised
    assert "94" not in serialised  # the count value


def test_build_log_record_no_secret():
    """A secret sentinel must never appear in a serialised record."""
    secret = "SUPER_SECRET_API_KEY_12345"
    report = make_report()
    record = build_log_record(question="q", report=report)
    serialised = json.dumps(record, default=str)
    assert secret not in serialised


def test_build_log_record_error_status():
    report = make_report(status="error", error="planning failed")
    record = build_log_record(question="q", report=report)
    assert record["status"] == "error"
    assert record["error"] == "planning failed"


# ---------------------------------------------------------------------------
# log_investigation / read_log
# ---------------------------------------------------------------------------
def test_log_investigation_appends_and_reads(tmp_path: Path):
    path = tmp_path / "nested" / "log.jsonl"
    report = make_report()
    record = log_investigation(
        path,
        question="q",
        report=report,
        dataset_label="sample",
        n_rows=10,
        n_columns=3,
        latency_ms=50,
    )
    assert path.exists()
    records = read_log(path)
    assert len(records) == 1
    assert records[0]["status"] == record["status"]
    assert records[0]["latency_ms"] == 50


def test_log_investigation_appends_multiple(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    report = make_report()
    log_investigation(path, question="q1", report=report)
    log_investigation(path, question="q2", report=report)
    assert len(read_log(path)) == 2


def test_read_log_missing_returns_empty(tmp_path: Path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_log_investigation_never_raises_on_bad_path(tmp_path: Path, monkeypatch):
    # Point at a path that cannot be a file's parent (a file used as a dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = blocker / "log.jsonl"
    report = make_report()
    # Must not raise; record is still returned.
    record = log_investigation(bad, question="q", report=report)
    assert record["status"] == "ok"


def test_default_log_path_is_ignored_location():
    assert str(DEFAULT_LOG_PATH).endswith("investigations.jsonl")
    assert "logs" in str(DEFAULT_LOG_PATH)
