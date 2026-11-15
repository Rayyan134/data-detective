"""Offline scoring of investigations against a fixed question set.

Day 4 adds a real evaluation layer on top of the Day 3 baseline. The Day 3
scorer could only ask "did the model pick the expected tool?", because it threw
the computed tables away. This module keeps them: a question carries a small,
declarative ``check`` describing the answer we expect, and the check is applied
to the **computed findings** of the run. That gives four independent things to
measure from a single pass:

* **tool accuracy**  -- did the plan call the tool we expected?
* **parameter accuracy** -- did it call it with the right arguments?
* **refusal accuracy** -- did it decline the questions no tool can answer?
* **answer accuracy** -- does the computed evidence actually contain the answer?

Because the tools are deterministic, answer accuracy can be measured from a
plan-only run (the tools still execute; only the narrative call is skipped), so
the whole suite costs one model call per question.

The same module also carries a :class:`RecordingClient`, which wraps the real
Gemini client and records per-call latency and token usage. That is how the
latency and cost notes required by Day 4 are produced without teaching the
investigator about billing.

Public API:
    load_questions(path)                 -> dict
    evaluate_question(question, report)  -> QuestionScore
    aggregate(scores, ...)               -> Metrics
    params_match(tool, expected, chosen) -> bool
    RecordingClient                      -> client wrapper that records usage
    estimate_cost(usage, price_in, price_out) -> float | None

Everything here is pure Python over an ``InvestigationReport``; it never calls
the model, so it is covered by offline unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.investigator import InvestigationReport
from src.tools import TOOL_REGISTRY

__all__ = [
    "CheckOutcome",
    "QuestionScore",
    "Metrics",
    "CallRecord",
    "RecordingClient",
    "load_questions",
    "params_match",
    "evaluate_checks",
    "evaluate_question",
    "aggregate",
    "estimate_cost",
    "sum_usage",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckOutcome:
    """Whether one answer check passed, plus a human-readable reason."""

    ok: bool
    detail: str


@dataclass
class QuestionScore:
    """Everything measurable about one question, without any model judgement.

    ``tool_ok``/``params_ok`` are ``None`` for a question that expects a
    refusal (there is no tool to score), and ``refusal_ok`` is ``None`` for a
    question that expects a tool. That keeps each accuracy denominator honest
    instead of counting a "not applicable" as a pass or a fail.
    """

    id: str
    question: str
    status: str
    expected_tool: str | None
    chosen_tool: str | None
    chosen_params: dict[str, Any]
    tool_ok: bool | None
    params_ok: bool | None
    refusal_ok: bool | None
    answer: CheckOutcome
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    interpreted: bool = False
    error: str | None = None

    @property
    def plan_ok(self) -> bool:
        """True when the *planning* stage was right for this question.

        For a supported question that means the right tool with the right
        parameters; for a refusal question it means it refused.
        """
        if self.expected_tool is None:
            return bool(self.refusal_ok)
        return bool(self.tool_ok and self.params_ok)

    @property
    def refused(self) -> bool:
        return self.status == "unsupported"


@dataclass
class Metrics:
    """Aggregate numbers over a whole evaluation run."""

    n: int
    tool_accuracy: float
    param_accuracy: float | None
    refusal_accuracy: float | None
    answer_accuracy: float
    plan_accuracy: float
    faithfulness: float | None
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_max: float
    prompt_tokens: int
    output_tokens: int
    thoughts_tokens: int
    total_tokens: int
    cost_usd: float | None
    repaired: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_questions(path: str | Path) -> dict[str, Any]:
    """Read the evaluation question set from JSON."""
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Parameter comparison
# ---------------------------------------------------------------------------
def params_match(expected_tool: str, expected: dict, chosen: dict) -> bool:
    """True when every expected parameter matches the chosen one.

    A parameter the plan omits is compared against the tool's declared default,
    because leaving an optional parameter out is a correct way to request it.
    """
    declared = TOOL_REGISTRY[expected_tool]["parameters"]
    for key, value in expected.items():
        if key in chosen:
            actual = chosen[key]
        else:
            spec = declared.get(key, {})
            if "default" not in spec:
                return False
            actual = spec["default"]
        if isinstance(value, list):
            if list(actual) != list(value):
                return False
        elif actual != value:
            return False
    return True


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------
def _tables(report: InvestigationReport) -> list[pd.DataFrame]:
    """Every computed table in the report, in step order."""
    return [step.result for step in report.findings if step.result is not None]


def _as_check_list(check: Any) -> list[dict]:
    if check is None:
        return []
    if isinstance(check, list):
        return [item for item in check if isinstance(item, dict)]
    if isinstance(check, dict):
        return [check]
    return []


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _compare_value(actual: Any, expected: Any, tolerance: float) -> CheckOutcome:
    """Compare a computed value with the expected one.

    Numbers are compared with an explicit tolerance (the tools round to three
    decimals, so an exact float match would be brittle); everything else is
    compared as text.
    """
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            if abs(float(actual) - float(expected)) <= tolerance:
                return CheckOutcome(True, f"value {_fmt(expected)} (tolerance {tolerance})")
        except (TypeError, ValueError):
            pass
        return CheckOutcome(False, f"expected {_fmt(expected)}, got {actual!r}")
    if str(actual).strip() == str(expected).strip():
        return CheckOutcome(True, f"value {expected!r}")
    return CheckOutcome(False, f"expected {expected!r}, got {actual!r}")


def _try_check(check: dict, table: pd.DataFrame) -> CheckOutcome | None:
    """Apply one check to one table, or return None when it does not fit.

    ``None`` means "this table cannot answer this check" (a required column is
    absent, for example), which lets the caller fall through to another step.
    It is deliberately different from a failed check.
    """
    kind = check.get("kind")

    if kind == "top_row":
        group, value = check.get("group_column"), check.get("value_column")
        if group not in table.columns or value not in table.columns or table.empty:
            return None
        numbers = pd.to_numeric(table[value], errors="coerce")
        if numbers.isna().all():
            return None
        index = numbers.idxmin() if check.get("mode", "max") == "min" else numbers.idxmax()
        return _compare_value(table.loc[index, group], check.get("expected"), 0.0)

    if kind == "cell":
        group, value = check.get("group_column"), check.get("value_column")
        if group not in table.columns or value not in table.columns:
            return None
        target = str(check.get("group_value"))
        rows = table[table[group].astype(str) == target]
        if rows.empty:
            return CheckOutcome(False, f"no row where {group}={target!r}")
        return _compare_value(
            rows.iloc[0][value], check.get("expected"), float(check.get("tolerance", 0.0))
        )

    if kind == "matrix_cell":
        row, column = check.get("row"), check.get("column")
        if column not in table.columns or row not in table.index:
            return None
        return _compare_value(
            table.loc[row, column], check.get("expected"), float(check.get("tolerance", 0.0))
        )

    if kind == "cell_value":
        column = check.get("column")
        if column not in table.columns or table.empty:
            return None
        actual = table.iloc[int(check.get("index", 0))][column]
        return _compare_value(actual, check.get("expected"), float(check.get("tolerance", 0.0)))

    if kind == "row_count":
        return _compare_value(
            len(table), check.get("expected"), float(check.get("tolerance", 0.0))
        )

    return None


def _evaluate_one(check: dict, report: InvestigationReport) -> CheckOutcome:
    """Evaluate one check against the report's computed tables."""
    if check.get("kind") == "refuse":
        refused = report.status == "unsupported" or (
            report.plan is not None and not report.plan.is_supported
        )
        if refused:
            return CheckOutcome(True, "refused as expected")
        return CheckOutcome(False, f"did not refuse (status={report.status})")

    for table in _tables(report):
        outcome = _try_check(check, table)
        if outcome is not None:
            return outcome
    return CheckOutcome(
        False, f"no computed result could satisfy the {check.get('kind')!r} check"
    )


def evaluate_checks(check: Any, report: InvestigationReport) -> CheckOutcome:
    """Evaluate a question's check(s). Every check in a list must pass."""
    checks = _as_check_list(check)
    if not checks:
        return CheckOutcome(False, "the question defines no check")
    details: list[str] = []
    ok = True
    for item in checks:
        outcome = _evaluate_one(item, report)
        ok = ok and outcome.ok
        details.append(outcome.detail)
    return CheckOutcome(ok, "; ".join(details))


# ---------------------------------------------------------------------------
# Scoring one question
# ---------------------------------------------------------------------------
def _select_step(plan, expected_tool: str | None):
    """Pick the plan step that best represents the answer.

    A plan may contain several steps. Prefer the step that uses the expected
    tool (so an extra exploratory step does not hide a correct call); otherwise
    fall back to the first step, which is what a human would read first.
    """
    if plan is None or not plan.is_supported or not plan.steps:
        return None
    for step in plan.steps:
        if expected_tool is not None and step.tool == expected_tool:
            return step
    return plan.steps[0]


def evaluate_question(
    question: dict[str, Any],
    report: InvestigationReport,
    *,
    latency_s: float = 0.0,
    usage: dict[str, Any] | None = None,
) -> QuestionScore:
    """Score one question against one report. Pure and offline."""
    expected_tool = question.get("expected_tool")
    expected_params = question.get("expected_params") or {}
    step = _select_step(report.plan, expected_tool)

    chosen_tool = step.tool if step is not None else None
    tool_ok: bool | None
    params_ok: bool | None
    refusal_ok: bool | None

    if expected_tool is None:
        tool_ok = None
        params_ok = None
        refusal_ok = report.status == "unsupported"
    else:
        tool_ok = chosen_tool == expected_tool
        params_ok = bool(step is not None and tool_ok and params_match(
            expected_tool, expected_params, step.params
        ))
        refusal_ok = None

    interpreted = bool(
        report.interpretation.answer or report.interpretation.interpretation
    )

    return QuestionScore(
        id=question["id"],
        question=question["question"],
        status=report.status,
        expected_tool=expected_tool,
        chosen_tool=chosen_tool,
        chosen_params=dict(step.params) if step is not None else {},
        tool_ok=tool_ok,
        params_ok=params_ok,
        refusal_ok=refusal_ok,
        answer=evaluate_checks(question.get("check"), report),
        latency_s=round(float(latency_s), 3),
        usage=dict(usage or report.usage or {}),
        repairs=list(report.plan.repairs) if report.plan else [],
        unverified=list(report.unverified_numbers),
        interpreted=interpreted,
        error=report.error,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _ratio(passed: int, total: int) -> float | None:
    return None if total == 0 else round(passed / total, 4)


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of a small list (no numpy needed)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sum_usage(usages: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Add up the token counts of several Gemini ``usageMetadata`` dicts."""
    prompt = output = thoughts = total = 0
    for usage in usages:
        prompt += int(usage.get("promptTokenCount", 0) or 0)
        output += int(usage.get("candidatesTokenCount", 0) or 0)
        thoughts += int(usage.get("thoughtsTokenCount", 0) or 0)
        total += int(usage.get("totalTokenCount", 0) or 0)
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": output,
        "thoughtsTokenCount": thoughts,
        "totalTokenCount": total,
    }


def estimate_cost(
    usage: dict[str, Any], price_in: float | None, price_out: float | None
) -> float | None:
    """Estimate the cost of one run in USD, or None when no prices are set.

    ``price_in``/``price_out`` are USD per million tokens. Output price is
    applied to both candidate and thinking tokens, which is how Gemini bills
    reasoning models. Passing None for either rate disables the estimate rather
    than silently assuming a price.
    """
    if price_in is None or price_out is None:
        return None
    prompt = int(usage.get("promptTokenCount", 0) or 0)
    generated = int(usage.get("candidatesTokenCount", 0) or 0) + int(
        usage.get("thoughtsTokenCount", 0) or 0
    )
    return round(prompt / 1_000_000 * price_in + generated / 1_000_000 * price_out, 6)


def aggregate(
    scores: list[QuestionScore],
    *,
    price_in: float | None = None,
    price_out: float | None = None,
) -> Metrics:
    """Fold per-question scores into the headline metrics.

    Each accuracy has its own denominator: parameters are only scored on
    questions that expect a tool, refusals only on questions that expect a
    refusal, and faithfulness only on questions that were actually interpreted.
    """
    n = len(scores)
    supported = [s for s in scores if s.expected_tool is not None]
    refusals = [s for s in scores if s.expected_tool is None]
    interpreted = [s for s in scores if s.interpreted]

    tool_passed = sum(1 for s in scores if s.tool_ok or s.refusal_ok)
    param_passed = sum(1 for s in supported if s.params_ok)
    refusal_passed = sum(1 for s in refusals if s.refusal_ok)
    answer_passed = sum(1 for s in scores if s.answer.ok)
    plan_passed = sum(1 for s in scores if s.plan_ok)
    faithful = sum(1 for s in interpreted if not s.unverified)

    latencies = [s.latency_s for s in scores if s.latency_s > 0]
    tokens = sum_usage([s.usage for s in scores])

    return Metrics(
        n=n,
        tool_accuracy=_ratio(tool_passed, n) or 0.0,
        param_accuracy=_ratio(param_passed, len(supported)),
        refusal_accuracy=_ratio(refusal_passed, len(refusals)),
        answer_accuracy=_ratio(answer_passed, n) or 0.0,
        plan_accuracy=_ratio(plan_passed, n) or 0.0,
        faithfulness=_ratio(faithful, len(interpreted)),
        latency_mean=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        latency_p50=round(_percentile(latencies, 0.50), 2),
        latency_p95=round(_percentile(latencies, 0.95), 2),
        latency_max=round(max(latencies), 2) if latencies else 0.0,
        prompt_tokens=tokens["promptTokenCount"],
        output_tokens=tokens["candidatesTokenCount"],
        thoughts_tokens=tokens["thoughtsTokenCount"],
        total_tokens=tokens["totalTokenCount"],
        cost_usd=estimate_cost(tokens, price_in, price_out),
        repaired=sum(1 for s in scores if s.repairs),
    )


# ---------------------------------------------------------------------------
# A client that records latency and tokens
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    """One model call: how long it took and what it cost in tokens."""

    latency_s: float
    usage: dict[str, Any]
    finish_reason: str | None
    model: str


class RecordingClient:
    """Wrap a client exposing ``generate_json`` and record every call.

    The investigator only sees the wrapped client, so latency and token usage
    are measured at the seam without changing the pipeline. On a retry the
    wrapper records both attempts, so the latency figures include the retry the
    user actually waited for.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[CallRecord] = []

    @property
    def model(self) -> str:
        return str(getattr(self.inner, "model", "") or "")

    def generate_json(self, prompt: str, **kwargs: Any) -> Any:
        started = time.perf_counter()
        response = self.inner.generate_json(prompt, **kwargs)
        elapsed = time.perf_counter() - started
        self.calls.append(
            CallRecord(
                latency_s=round(elapsed, 3),
                usage=dict(getattr(response, "usage", {}) or {}),
                finish_reason=getattr(response, "finish_reason", None),
                model=str(getattr(response, "model", "") or self.model),
            )
        )
        return response

    # -- convenience -------------------------------------------------------
    def reset(self) -> None:
        self.calls.clear()

    @property
    def total_latency_s(self) -> float:
        return round(sum(call.latency_s for call in self.calls), 3)

    def usage(self) -> dict[str, int]:
        return sum_usage(call.usage for call in self.calls)
