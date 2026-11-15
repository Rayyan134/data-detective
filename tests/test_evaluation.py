"""Tests for src.evaluation.

Everything here runs offline: reports are built either from hand-made result
tables or by running the real (deterministic) tools on the committed sample
dataset. No model and no API key are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import (
    RecordingClient,
    aggregate,
    estimate_cost,
    evaluate_checks,
    evaluate_question,
    load_questions,
    params_match,
    sum_usage,
)
from src.investigator import (
    InvestigationReport,
    PlanStep,
    StepResult,
    run_plan,
    validate_plan,
)
from src.profiler import load_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def report_with(*tables: pd.DataFrame, status: str = "ok") -> InvestigationReport:
    """A report whose findings are the given tables."""
    steps = [StepResult(tool="fake", result=table) for table in tables]
    return InvestigationReport(question="q", status=status, steps=steps)


def plan_report(df, tool: str, params: dict) -> InvestigationReport:
    """A real report produced by validating and running one plan step."""
    plan = validate_plan(
        {
            "status": "plan",
            "reasoning": "",
            "steps": [{"tool": tool, "params": params, "purpose": ""}],
        },
        df,
    )
    return InvestigationReport(
        question="q", status="ok", plan=plan, steps=run_plan(df, plan)
    )


def refused_report(df) -> InvestigationReport:
    plan = validate_plan(
        {
            "status": "unsupported",
            "reasoning": "",
            "unsupported_reason": "no tool can answer this",
        },
        df,
    )
    return InvestigationReport(question="q", status="unsupported", plan=plan)


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    return load_csv(PROJECT_ROOT / "data" / "sample_sales.csv")


@pytest.fixture(scope="module")
def exam() -> dict:
    return load_questions(PROJECT_ROOT / "eval" / "questions.json")


# ---------------------------------------------------------------------------
# params_match
# ---------------------------------------------------------------------------
def test_params_match_uses_declared_default_when_omitted():
    assert params_match("group_comparison", {"metric": "mean"}, {"group_column": "region"})


def test_params_match_rejects_a_different_value():
    assert not params_match(
        "group_comparison",
        {"group_column": "region"},
        {"group_column": "product"},
    )


def test_params_match_compares_lists_in_order():
    assert params_match("correlation", {"columns": ["a", "b"]}, {"columns": ["a", "b"]})
    assert not params_match("correlation", {"columns": ["a", "b"]}, {"columns": ["b", "a"]})


# ---------------------------------------------------------------------------
# Check primitives
# ---------------------------------------------------------------------------
def test_top_row_finds_the_max_group():
    table = pd.DataFrame({"region": ["A", "B"], "mean": [1.0, 5.0]})

    outcome = evaluate_checks(
        {"kind": "top_row", "group_column": "region", "value_column": "mean", "expected": "B"},
        report_with(table),
    )

    assert outcome.ok


def test_top_row_fails_when_the_wrong_group_wins():
    table = pd.DataFrame({"region": ["A", "B"], "mean": [1.0, 5.0]})

    outcome = evaluate_checks(
        {"kind": "top_row", "group_column": "region", "value_column": "mean", "expected": "A"},
        report_with(table),
    )

    assert not outcome.ok


def test_cell_reads_a_named_group():
    table = pd.DataFrame({"product": ["Pro", "Basic"], "sum": [49148.6, 39713.27]})

    outcome = evaluate_checks(
        {
            "kind": "cell",
            "group_column": "product",
            "group_value": "Pro",
            "value_column": "sum",
            "expected": 49148.6,
            "tolerance": 0.01,
        },
        report_with(table),
    )

    assert outcome.ok


def test_cell_fails_for_a_missing_group():
    table = pd.DataFrame({"product": ["Pro"], "sum": [1.0]})

    outcome = evaluate_checks(
        {
            "kind": "cell",
            "group_column": "product",
            "group_value": "Nope",
            "value_column": "sum",
            "expected": 1.0,
        },
        report_with(table),
    )

    assert not outcome.ok


def test_matrix_cell_reads_the_index_label():
    table = pd.DataFrame(
        {"discount_pct": [1.0, -0.118], "revenue": [-0.118, 1.0]},
        index=["discount_pct", "revenue"],
    )

    outcome = evaluate_checks(
        {"kind": "matrix_cell", "row": "discount_pct", "column": "revenue", "expected": -0.118, "tolerance": 0.005},
        report_with(table),
    )

    assert outcome.ok


def test_cell_value_reads_a_constant_column():
    table = pd.DataFrame({"revenue": [1.0, 2.0], "iqr_upper_bound": [2084.13, 2084.13]})

    outcome = evaluate_checks(
        {"kind": "cell_value", "column": "iqr_upper_bound", "expected": 2084.13, "tolerance": 0.01},
        report_with(table),
    )

    assert outcome.ok


def test_row_count_checks_the_number_of_rows():
    table = pd.DataFrame({"a": range(10)})

    assert evaluate_checks({"kind": "row_count", "expected": 10}, report_with(table)).ok
    assert not evaluate_checks({"kind": "row_count", "expected": 9}, report_with(table)).ok


def test_refuse_check_passes_only_on_a_refusal():
    df = pd.DataFrame({"a": [1]})

    assert evaluate_checks({"kind": "refuse"}, refused_report(df)).ok
    assert not evaluate_checks({"kind": "refuse"}, plan_report(df, "value_counts_table", {"column": "a"})).ok


def test_a_list_of_checks_requires_every_one():
    table = pd.DataFrame({"period": ["2025-07-31"], "total": [20024.78]})

    assert evaluate_checks(
        [
            {"kind": "row_count", "expected": 1},
            {"kind": "top_row", "group_column": "period", "value_column": "total", "expected": "2025-07-31"},
        ],
        report_with(table),
    ).ok

    assert not evaluate_checks(
        [
            {"kind": "row_count", "expected": 8},
            {"kind": "top_row", "group_column": "period", "value_column": "total", "expected": "2025-07-31"},
        ],
        report_with(table),
    ).ok


def test_a_check_that_cannot_be_applied_fails():
    # The table has none of the columns the check needs.
    outcome = evaluate_checks(
        {"kind": "top_row", "group_column": "region", "value_column": "mean", "expected": "East"},
        report_with(pd.DataFrame({"unrelated": [1]})),
    )

    assert not outcome.ok
    assert "no computed result" in outcome.detail


def test_check_falls_through_to_a_later_table():
    first = pd.DataFrame({"unrelated": [1]})
    second = pd.DataFrame({"region": ["East"], "mean": [912.007]})

    outcome = evaluate_checks(
        {"kind": "top_row", "group_column": "region", "value_column": "mean", "expected": "East"},
        report_with(first, second),
    )

    assert outcome.ok


# ---------------------------------------------------------------------------
# evaluate_question
# ---------------------------------------------------------------------------
def test_evaluate_question_scores_tool_params_and_answer(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q01")
    report = plan_report(
        sales,
        "group_comparison",
        {"group_column": "region", "value_column": "revenue", "metric": "mean"},
    )

    score = evaluate_question(question, report)

    assert score.tool_ok is True
    assert score.params_ok is True
    assert score.answer.ok
    assert score.plan_ok


def test_evaluate_question_flags_the_wrong_tool(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q01")
    report = plan_report(sales, "value_counts_table", {"column": "region"})

    score = evaluate_question(question, report)

    assert score.tool_ok is False
    assert score.params_ok is False
    assert not score.answer.ok


def test_evaluate_question_prefers_the_step_that_uses_the_expected_tool(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q01")
    plan = validate_plan(
        {
            "status": "plan",
            "reasoning": "",
            "steps": [
                {"tool": "value_counts_table", "params": {"column": "region"}},
                {
                    "tool": "group_comparison",
                    "params": {"group_column": "region", "value_column": "revenue", "metric": "mean"},
                },
            ],
        },
        sales,
    )
    report = InvestigationReport(
        question="q", status="ok", plan=plan, steps=run_plan(sales, plan)
    )

    score = evaluate_question(question, report)

    assert score.chosen_tool == "group_comparison"
    assert score.tool_ok and score.params_ok


def test_evaluate_question_scores_a_refusal(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q15")

    score = evaluate_question(question, refused_report(sales))

    assert score.refusal_ok is True
    assert score.tool_ok is None
    assert score.plan_ok


def test_evaluate_question_flags_a_missed_refusal(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q15")
    report = plan_report(sales, "value_counts_table", {"column": "region"})

    score = evaluate_question(question, report)

    assert score.refusal_ok is False
    assert not score.plan_ok


def test_evaluate_question_records_repairs(sales, exam):
    question = next(q for q in exam["questions"] if q["id"] == "Q01")
    plan = validate_plan(
        {
            "status": "plan",
            "reasoning": "",
            "steps": [
                {
                    "tool": "group_comparison",
                    "params": {
                        "group_column": "region",
                        "value_column": "revenue trailing text",
                        "metric": "mean",
                    },
                }
            ],
        },
        sales,
    )
    report = InvestigationReport(
        question="q", status="ok", plan=plan, steps=run_plan(sales, plan)
    )

    score = evaluate_question(question, report)

    assert score.repairs
    assert score.params_ok


# ---------------------------------------------------------------------------
# The exam set is self-consistent: the expected plan really does answer it
# ---------------------------------------------------------------------------
def test_every_supported_question_check_passes_on_its_expected_plan(sales, exam):
    for question in exam["questions"]:
        if question["expected_tool"] is None:
            continue
        report = plan_report(sales, question["expected_tool"], question["expected_params"])
        score = evaluate_question(question, report)
        assert score.tool_ok and score.params_ok, question["id"]
        assert score.answer.ok, f"{question['id']}: {score.answer.detail}"


def test_every_refusal_question_passes_on_a_refusal(sales, exam):
    for question in exam["questions"]:
        if question["expected_tool"] is not None:
            continue
        score = evaluate_question(question, refused_report(sales))
        assert score.refusal_ok, question["id"]
        assert score.answer.ok, f"{question['id']}: {score.answer.detail}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _score(question, report, **kwargs):
    return evaluate_question(question, report, **kwargs)


def test_aggregate_computes_each_accuracy_on_its_own_denominator(sales, exam):
    q01 = next(q for q in exam["questions"] if q["id"] == "Q01")
    q15 = next(q for q in exam["questions"] if q["id"] == "Q15")

    scores = [
        _score(q01, plan_report(sales, "group_comparison", {"group_column": "region", "value_column": "revenue", "metric": "mean"}), latency_s=2.0, usage={"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5, "totalTokenCount": 125}),
        _score(q15, refused_report(sales), latency_s=4.0, usage={"promptTokenCount": 50, "candidatesTokenCount": 10, "totalTokenCount": 60}),
    ]

    metrics = aggregate(scores)

    assert metrics.n == 2
    assert metrics.tool_accuracy == 1.0
    assert metrics.param_accuracy == 1.0          # only Q01 is scored
    assert metrics.refusal_accuracy == 1.0        # only Q15 is scored
    assert metrics.answer_accuracy == 1.0
    assert metrics.plan_accuracy == 1.0
    assert metrics.latency_mean == 3.0
    assert metrics.latency_max == 4.0
    assert metrics.prompt_tokens == 150
    assert metrics.total_tokens == 185
    assert metrics.cost_usd is None               # no prices configured


def test_aggregate_reports_none_when_a_denominator_is_empty(sales, exam):
    q01 = next(q for q in exam["questions"] if q["id"] == "Q01")
    scores = [
        _score(q01, plan_report(sales, "group_comparison", {"group_column": "region", "value_column": "revenue", "metric": "mean"}))
    ]

    metrics = aggregate(scores)

    assert metrics.refusal_accuracy is None
    assert metrics.faithfulness is None


def test_aggregate_counts_faithfulness_over_interpreted_questions(sales, exam):
    q01 = next(q for q in exam["questions"] if q["id"] == "Q01")
    report = plan_report(sales, "group_comparison", {"group_column": "region", "value_column": "revenue", "metric": "mean"})
    report.interpretation.answer = "East has the highest mean revenue of 912.007."
    report.unverified_numbers = []

    honest = evaluate_question(q01, report)
    assert honest.interpreted

    report.unverified_numbers = ["12345.67"]
    dishonest = evaluate_question(q01, report)

    assert aggregate([honest]).faithfulness == 1.0
    assert aggregate([honest, dishonest]).faithfulness == 0.5


def test_aggregate_counts_repairs(sales, exam):
    q01 = next(q for q in exam["questions"] if q["id"] == "Q01")
    plan = validate_plan(
        {
            "status": "plan",
            "reasoning": "",
            "steps": [
                {
                    "tool": "group_comparison",
                    "params": {"group_column": "region", "value_column": "revenue junk", "metric": "mean"},
                }
            ],
        },
        sales,
    )
    report = InvestigationReport(question="q", status="ok", plan=plan, steps=run_plan(sales, plan))

    assert aggregate([evaluate_question(q01, report)]).repaired == 1


# ---------------------------------------------------------------------------
# Token and cost helpers
# ---------------------------------------------------------------------------
def test_sum_usage_adds_the_gemini_token_fields():
    totals = sum_usage(
        [
            {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 15},
            {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
        ]
    )

    assert totals == {
        "promptTokenCount": 15,
        "candidatesTokenCount": 3,
        "thoughtsTokenCount": 3,
        "totalTokenCount": 21,
    }


def test_estimate_cost_is_none_without_prices():
    assert estimate_cost({"promptTokenCount": 1000}, None, None) is None


def test_estimate_cost_charges_thinking_tokens_as_output():
    usage = {"promptTokenCount": 1_000_000, "candidatesTokenCount": 500_000, "thoughtsTokenCount": 500_000}

    cost = estimate_cost(usage, price_in=0.30, price_out=2.50)

    assert cost == pytest.approx(0.30 + 1_000_000 / 1_000_000 * 2.50)


# ---------------------------------------------------------------------------
# RecordingClient
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, usage):
        self.usage = usage
        self.finish_reason = "STOP"
        self.model = "fake-model"


class _FakeInner:
    model = "fake-model"

    def __init__(self):
        self.prompts = []

    def generate_json(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return _FakeResponse({"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10})


def test_recording_client_records_latency_and_usage():
    inner = _FakeInner()
    client = RecordingClient(inner)

    client.generate_json("hello")
    client.generate_json("again")

    assert len(client.calls) == 2
    assert client.model == "fake-model"
    assert client.calls[0].finish_reason == "STOP"
    assert client.usage() == {
        "promptTokenCount": 14,
        "candidatesTokenCount": 6,
        "thoughtsTokenCount": 0,
        "totalTokenCount": 20,
    }
    assert client.total_latency_s >= 0

    client.reset()
    assert client.calls == []
