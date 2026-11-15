"""Tests for src.investigator.

Everything here runs offline. A ``FakeClient`` stands in for the Gemini client
and replays canned JSON, so the planning, validation, execution and reporting
paths are all exercised without a network or an API key. Run with:

    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.investigator import (
    DEFAULT_THINKING_LEVEL,
    INTERPRETATION_SCHEMA,
    MAX_STEPS,
    PLAN_SCHEMA,
    PlanError,
    build_dataset_context,
    build_evidence,
    build_interpretation_prompt,
    build_plan_prompt,
    build_tool_catalogue,
    find_unverified_numbers,
    format_result_table,
    investigate,
    plan_investigation,
    run_plan,
    validate_plan,
)
from src.llm_client import LLMError, LLMTimeoutError
from src.profiler import load_csv
from src.tools import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeResponse:
    """Stands in for ``src.llm_client.LLMResponse``."""

    def __init__(self, payload, model="fake-model", usage=None):
        self._payload = payload
        self.text = json.dumps(payload)
        self.model = model
        self.usage = usage or {}

    def json(self):
        return self._payload


class FakeClient:
    """Replays canned responses and records the prompts it was handed.

    The last response repeats if more calls arrive, and an entry that is an
    ``Exception`` is raised instead of returned, which is how the error and
    fallback paths are exercised.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_json(
        self,
        prompt,
        *,
        schema=None,
        system_instruction=None,
        temperature=0.0,
        max_output_tokens=None,
        thinking_level=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system_instruction": system_instruction,
                "thinking_level": thinking_level,
            }
        )
        if not self._responses:
            raise AssertionError("FakeClient ran out of canned responses.")
        item = (
            self._responses.pop(0)
            if len(self._responses) > 1
            else self._responses[0]
        )
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def step(tool: str, **params):
    return {"tool": tool, "params": params, "purpose": f"run {tool}"}


def plan_payload(*steps, status="plan", reasoning="a reason", unsupported_reason=""):
    return {
        "status": status,
        "reasoning": reasoning,
        "unsupported_reason": unsupported_reason,
        "steps": list(steps),
    }


def interp_payload(
    answer="East", interpretation="East leads.", caveats=None, confidence="high"
):
    return {
        "answer": answer,
        "interpretation": interpretation,
        "caveats": caveats or [],
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def df() -> pd.DataFrame:
    """A small dataset shaped like data/sample_sales.csv."""
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{index:04d}" for index in range(1, 9)],
            "order_date": [
                "2025-01-01", "2025-01-15", "2025-02-01", "2025-02-15",
                "2025-03-01", "2025-03-15", "2025-04-01", "2025-04-15",
            ],
            "region": ["East", "North", "East", "North", "South", "South", "East", "North"],
            "product": ["Basic", "Pro", "Basic", "Pro", "Enterprise", "Basic", "Pro", "Enterprise"],
            "channel": ["Online", "Retail", "Online", "Retail", "Partner", "Online", "Retail", "Online"],
            "units": [10, 20, 30, 40, 5, 15, 25, 35],
            "revenue": [100.0, 250.0, 300.0, 420.0, 50.0, 150.0, 260.0, 380.0],
            "satisfaction": [5.0, None, 3.0, 4.0, None, 5.0, 4.0, None],
        }
    )


# ---------------------------------------------------------------------------
# Dataset context
# ---------------------------------------------------------------------------
def test_context_lists_columns_types_and_categories(df):
    context = build_dataset_context(df)

    assert "region | text" in context
    assert "values: East, North, South" in context
    assert "revenue | numeric" in context
    assert "order_date | text (date-like)" in context


def test_context_includes_numeric_statistics(df):
    context = build_dataset_context(df)

    # units mean = (10+20+30+40+5+15+25+35) / 8 = 22.5
    assert "mean 22.5" in context
    assert "max 420" in context


def test_context_reports_missingness(df):
    context = build_dataset_context(df)

    # satisfaction has 3 missing of 8 rows = 37.5%
    assert "missing 37.5% (3 of 8)" in context


def test_context_caps_examples_instead_of_dumping_columns(df):
    context = build_dataset_context(df)

    # order_id has 8 values; only a few examples should be shown.
    assert "ORD-0001" in context
    assert "ORD-0004" not in context


def test_context_rejects_a_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        build_dataset_context([1, 2, 3])


# ---------------------------------------------------------------------------
# Tool catalogue and prompts
# ---------------------------------------------------------------------------
def test_tool_catalogue_matches_the_registry():
    catalogue = build_tool_catalogue()

    assert [tool["name"] for tool in catalogue] == list(TOOL_REGISTRY)


def test_plan_prompt_contains_question_tools_and_columns(df):
    prompt = build_plan_prompt("Which region earns most?", df)

    assert "Which region earns most?" in prompt
    assert "value_counts_table" in prompt
    assert "region" in prompt


def test_plan_prompt_never_contains_the_whole_column(df):
    prompt = build_plan_prompt("anything", df)

    assert "ORD-0004" not in prompt


def test_plan_schema_restricts_tools_to_the_registry():
    tool_enum = PLAN_SCHEMA["properties"]["steps"]["items"]["properties"]["tool"]["enum"]

    assert tool_enum == list(TOOL_REGISTRY)


def test_schemas_are_json_serialisable():
    assert json.dumps(PLAN_SCHEMA)
    assert json.dumps(INTERPRETATION_SCHEMA)


def test_interpretation_schema_requires_core_fields():
    assert set(INTERPRETATION_SCHEMA["required"]) == {
        "answer",
        "interpretation",
        "confidence",
    }


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------
def test_valid_plan_is_accepted(df):
    plan = validate_plan(
        plan_payload(
            step("group_comparison", group_column="region", value_column="revenue"),
        ),
        df,
    )

    assert plan.is_supported
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "group_comparison"
    assert plan.steps[0].params == {"group_column": "region", "value_column": "revenue"}
    assert plan.steps[0].purpose == "run group_comparison"


def test_plan_must_be_an_object(df):
    with pytest.raises(PlanError, match="JSON object"):
        validate_plan(["not", "an", "object"], df)


def test_plan_rejects_unknown_status(df):
    with pytest.raises(PlanError, match="status"):
        validate_plan(plan_payload(status="maybe"), df)


def test_plan_rejects_unknown_tool(df):
    payload = plan_payload({"tool": "delete_everything", "params": {}})

    with pytest.raises(PlanError, match="unknown tool"):
        validate_plan(payload, df)


def test_plan_rejects_unknown_parameter(df):
    payload = plan_payload(step("value_counts_table", column="region", colour="red"))

    with pytest.raises(PlanError, match="unknown parameter 'colour'"):
        validate_plan(payload, df)


def test_plan_rejects_missing_required_parameter(df):
    payload = plan_payload(step("group_comparison", group_column="region"))

    with pytest.raises(PlanError, match="missing required parameter"):
        validate_plan(payload, df)


def test_plan_rejects_an_unknown_column(df):
    payload = plan_payload(step("value_counts_table", column="country"))

    with pytest.raises(PlanError, match="Available columns"):
        validate_plan(payload, df)


def test_plan_rejects_an_unknown_column_inside_a_list(df):
    payload = plan_payload(step("correlation", columns=["revenue", "profit"]))

    with pytest.raises(PlanError, match="unknown column 'profit'"):
        validate_plan(payload, df)


def test_plan_rejects_a_non_integer_for_an_integer_parameter(df):
    payload = plan_payload(step("value_counts_table", column="region", top_n=2.5))

    with pytest.raises(PlanError, match="whole number"):
        validate_plan(payload, df)


def test_plan_rejects_a_string_for_an_integer_parameter(df):
    payload = plan_payload(step("value_counts_table", column="region", top_n="ten"))

    with pytest.raises(PlanError, match="integer"):
        validate_plan(payload, df)


def test_plan_rejects_an_invalid_enum_value(df):
    payload = plan_payload(
        step("group_comparison", group_column="region", value_column="revenue", metric="mode")
    )

    with pytest.raises(PlanError, match="must be one of"):
        validate_plan(payload, df)


def test_plan_ignores_explicit_nulls(df):
    payload = plan_payload(
        step("value_counts_table", column="region", top_n=None)
    )

    plan = validate_plan(payload, df)

    assert plan.steps[0].params == {"column": "region"}


def test_plan_with_no_steps_is_rejected(df):
    with pytest.raises(PlanError, match="at least one step"):
        validate_plan(plan_payload(), df)


def test_plan_over_the_step_limit_is_rejected(df):
    steps = [step("value_counts_table", column="region") for _ in range(MAX_STEPS + 1)]

    with pytest.raises(PlanError, match="more than the limit"):
        validate_plan(plan_payload(*steps), df)


def test_steps_must_be_a_list(df):
    payload = {"status": "plan", "reasoning": "", "steps": "nope"}

    with pytest.raises(PlanError, match="must be a list"):
        validate_plan(payload, df)


def test_unsupported_plan_is_kept_with_its_reason(df):
    payload = plan_payload(status="unsupported", unsupported_reason="Needs a segment column.")

    plan = validate_plan(payload, df)

    assert not plan.is_supported
    assert plan.steps == []
    assert plan.unsupported_reason == "Needs a segment column."


def test_unsupported_plan_gets_a_default_reason(df):
    payload = plan_payload(status="unsupported")

    plan = validate_plan(payload, df)

    assert plan.unsupported_reason


# ---------------------------------------------------------------------------
# Parameter normalisation (model leaks trailing text into string fields)
# ---------------------------------------------------------------------------
def test_plan_normalises_trailing_text_in_a_column_parameter(df):
    payload = plan_payload(
        step("value_counts_table", column="region metric: mean please confirm")
    )

    plan = validate_plan(payload, df)

    assert plan.steps[0].params["column"] == "region"
    assert plan.repairs and "normalised" in plan.repairs[0]


def test_plan_normalises_trailing_text_when_only_one_column_can_match(df):
    # No boundary, but "revenue" is the only known column that fits.
    payload = plan_payload(step("value_counts_table", column="revenuepopular"))

    plan = validate_plan(payload, df)

    assert plan.steps[0].params["column"] == "revenue"
    assert plan.repairs


def test_plan_normalises_trailing_text_in_an_enum_parameter(df):
    payload = plan_payload(
        step("group_comparison", group_column="region", value_column="revenue", metric="mean ok")
    )

    plan = validate_plan(payload, df)

    assert plan.steps[0].params["metric"] == "mean"
    assert plan.repairs


def test_plan_normalises_trailing_text_inside_a_column_list(df):
    payload = plan_payload(step("correlation", columns=["revenue junk", "units"]))

    plan = validate_plan(payload, df)

    assert plan.steps[0].params["columns"] == ["revenue", "units"]
    assert plan.repairs


def test_plan_still_rejects_a_genuinely_unknown_column(df):
    payload = plan_payload(step("value_counts_table", column="profit"))

    with pytest.raises(PlanError, match="unknown column 'profit'"):
        validate_plan(payload, df)


def test_plan_still_rejects_a_genuinely_wrong_enum(df):
    payload = plan_payload(
        step("group_comparison", group_column="region", value_column="revenue", metric="mode")
    )

    with pytest.raises(PlanError, match="must be one of"):
        validate_plan(payload, df)


def test_plan_without_leaked_text_records_no_repairs(df):
    payload = plan_payload(step("value_counts_table", column="region"))

    plan = validate_plan(payload, df)

    assert plan.repairs == []


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def test_run_plan_executes_each_step(df):
    plan = validate_plan(
        plan_payload(step("group_comparison", group_column="region", value_column="revenue")),
        df,
    )

    results = run_plan(df, plan)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].n_rows == 3


def test_run_plan_records_a_failed_tool_without_raising(df):
    # correlation with a single column validates but fails inside the tool.
    plan = validate_plan(plan_payload(step("correlation", columns=["revenue"])), df)

    results = run_plan(df, plan)

    assert not results[0].ok
    assert "at least two" in results[0].error


def test_run_plan_captures_unexpected_errors(monkeypatch, df):
    def boom(*args, **kwargs):
        raise RuntimeError("exploded")

    monkeypatch.setattr("src.investigator.run_tool", boom)
    plan = validate_plan(plan_payload(step("value_counts_table", column="region")), df)

    results = run_plan(df, plan)

    assert not results[0].ok
    assert "Unexpected error" in results[0].error


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------
def test_format_result_table_handles_no_result():
    assert format_result_table(None) == "_(no result)_"


def test_format_result_table_handles_no_rows(df):
    assert format_result_table(df.iloc[0:0]) == "_(no rows returned)_"


def test_format_result_table_renders_markdown(df):
    table = format_result_table(df[["region", "revenue"]].head(2))

    assert table.splitlines()[0] == "| region | revenue |"
    assert "East" in table


def test_format_result_table_caps_rows(df):
    table = format_result_table(df, max_rows=3)

    assert "showing the first 3 of 8 rows" in table


def test_build_evidence_includes_steps_and_results(df):
    plan = validate_plan(
        plan_payload(step("value_counts_table", column="region")), df
    )
    steps = run_plan(df, plan)

    evidence = build_evidence(steps)

    assert "STEP 1 — value_counts_table" in evidence
    assert '"column": "region"' in evidence
    assert "East" in evidence


def test_build_evidence_marks_failures(df):
    plan = validate_plan(plan_payload(step("correlation", columns=["revenue"])), df)
    steps = run_plan(df, plan)

    evidence = build_evidence(steps)

    assert "FAILED" in evidence


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------
def test_unverified_numbers_flags_an_invented_statistic():
    flagged = find_unverified_numbers(
        "Revenue averaged 999.99 per order.", "revenue mean 220.0"
    )

    assert flagged == ["999.99"]


def test_unverified_numbers_accepts_an_exact_match():
    assert find_unverified_numbers("r = 0.703", "units revenue 0.703") == []


def test_unverified_numbers_accepts_rounding():
    # 912.007 rounded to two decimals is not a hallucination.
    assert find_unverified_numbers("mean of 912.01", "mean 912.007") == []


def test_unverified_numbers_accepts_thousands_separators():
    assert find_unverified_numbers("total 20,024.78", "total 20024.78") == []


def test_unverified_numbers_ignores_small_counts():
    assert find_unverified_numbers("across 3 regions", "") == []


def test_unverified_numbers_is_sign_insensitive():
    assert find_unverified_numbers("fell 32.1%", "pct_change -32.1") == []


def test_unverified_numbers_accepts_a_percent_fraction():
    # The model may express "2.8%" as the fraction "0.028".
    assert find_unverified_numbers("a change of 0.028", "pct_change 2.8") == []


# ---------------------------------------------------------------------------
# plan_investigation
# ---------------------------------------------------------------------------
def test_plan_investigation_returns_a_validated_plan(df):
    client = FakeClient(
        plan_payload(step("value_counts_table", column="channel"))
    )

    plan = plan_investigation("Which channel is most common?", df, client=client)

    assert plan.is_supported
    assert plan.steps[0].tool == "value_counts_table"


def test_plan_investigation_uses_the_default_thinking_level(df):
    client = FakeClient(plan_payload(step("value_counts_table", column="channel")))

    plan_investigation("Which channel is most common?", df, client=client)

    assert client.calls[0]["thinking_level"] == DEFAULT_THINKING_LEVEL
    assert client.calls[0]["schema"] is PLAN_SCHEMA


def test_plan_investigation_raises_on_an_invalid_plan(df):
    client = FakeClient(plan_payload({"tool": "nope", "params": {}}))

    with pytest.raises(PlanError):
        plan_investigation("anything", df, client=client)


# ---------------------------------------------------------------------------
# investigate: the graceful end-to-end entry point
# ---------------------------------------------------------------------------
def test_investigate_produces_evidence_and_an_interpretation(df):
    client = FakeClient(
        plan_payload(
            step("group_comparison", group_column="region", value_column="revenue", metric="mean")
        ),
        interp_payload(answer="East", interpretation="East leads at 220."),
    )

    report = investigate("Which region earns most?", df, client=client)

    assert report.status == "ok"
    assert len(report.findings) == 1
    assert report.findings[0].n_rows == 3
    assert report.interpretation.answer == "East"
    assert report.model == "fake-model"
    assert report.unverified_numbers == []


def test_investigate_feeds_real_numbers_into_the_interpretation_prompt(df):
    client = FakeClient(
        plan_payload(
            step("group_comparison", group_column="region", value_column="revenue", metric="mean")
        ),
        interp_payload(),
    )

    investigate("Which region earns most?", df, client=client)

    # East mean = (100 + 300 + 260) / 3 = 220.0, computed by the tool.
    assert "220" in client.calls[1]["prompt"]


def test_investigate_report_keeps_evidence_and_interpretation_apart(df):
    client = FakeClient(
        plan_payload(step("value_counts_table", column="region")),
        interp_payload(answer="East", interpretation="East is most common."),
    )

    report = investigate("Which region is most common?", df, client=client)
    markdown = report.to_markdown()

    assert "Findings (computed in Python)" in markdown
    assert "Answer (AI interpretation" in markdown


def test_investigate_surfaces_normalised_parameters(df):
    client = FakeClient(
        plan_payload(step("value_counts_table", column="region trailing text")),
        interp_payload(),
    )

    report = investigate("Which region is most common?", df, client=client)

    assert report.status == "ok"
    assert report.plan.repairs
    assert "Normalised parameters" in report.to_markdown()


def test_investigate_returns_unsupported_without_running_tools(df):
    client = FakeClient(
        plan_payload(status="unsupported", unsupported_reason="No segment column exists.")
    )

    report = investigate("Which segment is most profitable?", df, client=client)

    assert report.status == "unsupported"
    assert report.steps == []
    assert report.error is None
    assert report.plan.unsupported_reason == "No segment column exists."
    assert len(client.calls) == 1


def test_investigate_handles_a_planning_api_error(df):
    client = FakeClient(LLMError("quota exhausted"))

    report = investigate("anything", df, client=client)

    assert report.status == "error"
    assert "planning request failed" in report.error


def test_investigate_handles_a_rejected_plan(df):
    client = FakeClient(plan_payload({"tool": "delete_everything", "params": {}}))

    report = investigate("anything", df, client=client)

    assert report.status == "error"
    assert "rejected" in report.error


def test_investigate_handles_a_non_object_plan(df):
    client = FakeClient("not a plan")

    report = investigate("anything", df, client=client)

    assert report.status == "error"
    assert "rejected" in report.error


def test_investigate_handles_every_tool_failing(df):
    client = FakeClient(plan_payload(step("correlation", columns=["revenue"])))

    report = investigate("Are two things related?", df, client=client)

    assert report.status == "error"
    assert "Every tool call failed" in report.error
    assert len(client.calls) == 1  # no interpretation call was made


def test_investigate_keeps_findings_when_interpretation_fails(df):
    client = FakeClient(
        plan_payload(step("value_counts_table", column="region")),
        LLMError("reporting service down"),
    )

    report = investigate("Which region is most common?", df, client=client)

    assert report.status == "partial"
    assert len(report.findings) == 1
    assert "interpretation failed" in report.error


def test_investigate_can_skip_the_interpretation(df):
    client = FakeClient(plan_payload(step("value_counts_table", column="region")))

    report = investigate("Which region is most common?", df, client=client, interpret=False)

    assert report.status == "ok"
    assert report.interpretation.answer == ""
    assert len(client.calls) == 1


def test_investigate_falls_back_to_lower_thinking_on_timeout(df):
    client = FakeClient(
        LLMTimeoutError("too slow"),
        plan_payload(step("value_counts_table", column="region")),
        interp_payload(),
    )

    report = investigate("Which region is most common?", df, client=client)

    assert report.status == "ok"
    assert client.calls[0]["thinking_level"] == DEFAULT_THINKING_LEVEL
    assert client.calls[1]["thinking_level"] == "LOW"


def test_investigate_reports_a_timeout_when_fallback_is_disabled(df):
    client = FakeClient(LLMTimeoutError("too slow"))

    report = investigate(
        "Which region is most common?", df, client=client, fallback_thinking_level=None
    )

    assert report.status == "error"
    assert len(client.calls) == 1


def test_investigate_rejects_an_empty_question(df):
    client = FakeClient(plan_payload(step("value_counts_table", column="region")))

    report = investigate("   ", df, client=client)

    assert report.status == "error"
    assert client.calls == []


def test_investigate_rejects_a_non_dataframe():
    client = FakeClient(plan_payload(step("value_counts_table", column="region")))

    report = investigate("anything", "not a dataframe", client=client)

    assert report.status == "error"
    assert client.calls == []


def test_investigate_rejects_an_empty_dataset():
    empty = pd.DataFrame({"region": []})
    client = FakeClient(plan_payload(step("value_counts_table", column="region")))

    report = investigate("anything", empty, client=client)

    assert report.status == "error"
    assert client.calls == []


def test_investigate_without_a_key_reports_the_missing_key(df, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    report = investigate("anything", df)

    assert report.status == "error"
    assert "API key" in report.error


# ---------------------------------------------------------------------------
# Integration with the real sample dataset
# ---------------------------------------------------------------------------
def test_investigate_validates_against_the_real_sample_dataset():
    dataset = load_csv("data/sample_sales.csv")
    client = FakeClient(
        plan_payload(
            step("group_comparison", group_column="region", value_column="revenue", metric="mean")
        ),
        interp_payload(),
    )

    report = investigate("Which region has the highest average revenue?", dataset, client=client)

    assert report.status == "ok"
    assert report.findings[0].n_rows == 4


def test_interpretation_prompt_includes_dataset_and_evidence(df):
    plan = validate_plan(
        plan_payload(step("value_counts_table", column="region")), df
    )
    steps = run_plan(df, plan)

    prompt = build_interpretation_prompt("Which region is most common?", df, steps)

    assert "EVIDENCE" in prompt
    assert "Which region is most common?" in prompt
    assert "East" in prompt
