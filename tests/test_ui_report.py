"""Offline tests for src.ui_report (presentation helpers, no Streamlit)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.investigator import InvestigationReport, StepResult
from src.ui_report import (
    STATUS_STYLES,
    dataset_label,
    example_questions,
    finding_sections,
    interpretation_section,
    plan_lines,
    redaction_candidates,
    status_banner,
    unverified_warning,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(20)],
            "region": ["East", "West", "North", "South"] * 5,
            "product": ["Basic", "Pro", "Basic"] + ["Pro"] * 17,
            "revenue": [100.0, 200.0, 150.0, 300.0] * 5,
            "order_date": pd.date_range("2025-01-01", periods=20, freq="D"),
        }
    )


def make_people_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"] * 4,
            "age": [25, 30, 35] * 4,
            "city": ["NY", "LA", "SF"] * 4,
        }
    )


def report_with_status(status: str, error: str | None = None) -> InvestigationReport:
    return InvestigationReport(question="q", status=status, error=error)


# ---------------------------------------------------------------------------
# status_banner
# ---------------------------------------------------------------------------
def test_status_banner_levels():
    assert status_banner(report_with_status("ok"))["level"] == "success"
    assert status_banner(report_with_status("partial", "boom"))["level"] == "warning"
    assert status_banner(report_with_status("unsupported"))["level"] == "info"
    assert status_banner(report_with_status("error", "boom"))["level"] == "error"


def test_status_banner_uses_error_message():
    banner = status_banner(report_with_status("error", "planning failed"))
    assert banner["message"] == "planning failed"


def test_status_banner_unsupported_falls_back():
    banner = status_banner(report_with_status("unsupported"))
    assert "No available tool" in banner["message"] or "cannot be answered" in banner["message"]


def test_status_styles_covers_all_statuses():
    for status in ("ok", "partial", "unsupported", "error"):
        assert status in STATUS_STYLES


# ---------------------------------------------------------------------------
# plan_lines
# ---------------------------------------------------------------------------
def test_plan_lines_none():
    assert plan_lines(InvestigationReport(question="q", status="error")) == []


def test_plan_lines_unsupported():
    from src.investigator import InvestigationPlan

    plan = InvestigationPlan(status="unsupported", unsupported_reason="no segment column")
    report = InvestigationReport(question="q", status="unsupported", plan=plan)
    lines = plan_lines(report)
    assert len(lines) == 1
    assert "no segment column" in lines[0]


# ---------------------------------------------------------------------------
# finding_sections
# ---------------------------------------------------------------------------
def test_finding_sections_empty():
    report = InvestigationReport(question="q", status="ok")
    assert finding_sections(report) == []


def test_finding_sections_keys_and_frame():
    frame = pd.DataFrame({"a": [1, 2]})
    step = StepResult(tool="value_counts_table", params={"column": "region"}, result=frame)
    report = InvestigationReport(question="q", status="ok", steps=[step])
    sections = finding_sections(report)
    assert len(sections) == 1
    section = sections[0]
    assert section["title"] == "Step 1 — value_counts_table"
    assert section["frame"] is frame
    assert section["error"] is None
    assert isinstance(section["table"], str) and section["table"]


def test_finding_sections_failed_step():
    step = StepResult(tool="correlation", params={}, error="boom")
    report = InvestigationReport(question="q", status="error", steps=[step])
    sections = finding_sections(report)
    assert sections[0]["frame"] is None
    assert sections[0]["error"] == "boom"


# ---------------------------------------------------------------------------
# interpretation_section
# ---------------------------------------------------------------------------
def test_interpretation_section_none_when_empty():
    report = InvestigationReport(question="q", status="ok")
    assert interpretation_section(report) is None


def test_interpretation_section_returns_fields():
    from src.investigator import Interpretation

    report = InvestigationReport(
        question="q",
        status="ok",
        interpretation=Interpretation(answer="A", interpretation="B", confidence="high", caveats=["c"]),
    )
    section = interpretation_section(report)
    assert section["answer"] == "A"
    assert section["interpretation"] == "B"
    assert section["confidence"] == "high"
    assert section["caveats"] == ["c"]


# ---------------------------------------------------------------------------
# unverified_warning
# ---------------------------------------------------------------------------
def test_unverified_warning_none():
    assert unverified_warning(InvestigationReport(question="q", status="ok")) is None


def test_unverified_warning_lists_numbers():
    report = InvestigationReport(question="q", status="ok", unverified_numbers=["42", "3.14"])
    warning = unverified_warning(report)
    assert warning is not None
    assert "`42`" in warning
    assert "`3.14`" in warning


# ---------------------------------------------------------------------------
# redaction_candidates
# ---------------------------------------------------------------------------
def test_redaction_candidates_finds_id_like():
    df = make_sales_df()
    candidates = redaction_candidates(df)
    assert "order_id" in candidates


def test_redaction_candidates_high_cardinality_text():
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(50)],
            "amount": [1.0] * 50,
        }
    )
    candidates = redaction_candidates(df)
    assert "customer_id" in candidates


def test_redaction_candidates_empty_df():
    assert redaction_candidates(pd.DataFrame()) == []


def test_redaction_candidates_generalises():
    df = make_people_df()
    # "name" is a PII hint; age/city should not be suggested.
    candidates = redaction_candidates(df)
    assert "name" in candidates
    assert "age" not in candidates
    assert "city" not in candidates


# ---------------------------------------------------------------------------
# example_questions
# ---------------------------------------------------------------------------
def test_example_questions_sales():
    df = make_sales_df()
    questions = example_questions(df)
    assert questions
    assert any("revenue" in q and "region" in q for q in questions)


def test_example_questions_people():
    df = make_people_df()
    questions = example_questions(df)
    assert questions
    # No revenue/column-specific leakage; only generic templates.
    assert all(isinstance(q, str) and q for q in questions)


def test_example_questions_empty():
    assert example_questions(pd.DataFrame()) == []


def test_example_questions_generalises_to_any_csv():
    df = pd.DataFrame(
        {
            "temperature": [1.0, 2.0, 3.0, 4.0],
            "humidity": [10.0, 20.0, 30.0, 40.0],
            "station": ["A", "B", "A", "B"],
        }
    )
    questions = example_questions(df)
    assert any("temperature" in q for q in questions)
    assert any("station" in q for q in questions)


# ---------------------------------------------------------------------------
# dataset_label
# ---------------------------------------------------------------------------
def test_dataset_label():
    df = make_sales_df()
    label = dataset_label("sales.csv", df)
    assert "sales.csv" in label
    assert f"{len(df):,} rows" in label
    assert f"{df.shape[1]:,} columns" in label
