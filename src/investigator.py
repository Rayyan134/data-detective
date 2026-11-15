"""Agent planning, tool execution and evidence-based reporting.

This is the layer that turns a natural-language question into a sequence of
predefined tool calls, runs those tools in Python, and assembles a report that
keeps calculated evidence separate from the model's interpretation.

The model is never allowed to run code. It chooses a tool name from a fixed
catalogue and supplies parameters. Every parameter is validated against the
tool's declared schema *and* against the real columns of the dataset before
anything is executed, and execution goes through ``src.tools.run_tool`` only.

The pipeline has one seam: an object exposing ``generate_json(prompt, ...)``.
In production that is ``src.llm_client.GeminiClient``; in tests it is a fake,
so the whole module is testable without a network or an API key.

Public API:
    investigate(question, df, ...)      -> InvestigationReport   (never raises)
    plan_investigation(question, ...)   -> InvestigationPlan
    validate_plan(raw_plan, df)         -> InvestigationPlan
    run_plan(df, plan)                  -> list[StepResult]
    build_dataset_context(df, profile)  -> str
    build_tool_catalogue()              -> list[dict]
    build_evidence(steps)               -> str
    find_unverified_numbers(text, ev)   -> list[str]
    format_result_table(df, max_rows)   -> str

Exceptions:
    InvestigationError  -- base class for planning problems
    PlanError           -- the model's plan was rejected by validation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.llm_client import (
    GeminiClient,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
)
from src.profiler import numeric_summary, profile_dataset
from src.tools import TOOL_REGISTRY, ToolError, list_tools, run_tool

__all__ = [
    "InvestigationError",
    "PlanError",
    "PlanStep",
    "InvestigationPlan",
    "StepResult",
    "Interpretation",
    "InvestigationReport",
    "PLAN_SCHEMA",
    "INTERPRETATION_SCHEMA",
    "MAX_STEPS",
    "MAX_PREVIEW_ROWS",
    "DEFAULT_THINKING_LEVEL",
    "FALLBACK_THINKING_LEVEL",
    "build_dataset_context",
    "build_tool_catalogue",
    "build_plan_prompt",
    "build_evidence",
    "format_result_table",
    "find_unverified_numbers",
    "plan_investigation",
    "validate_plan",
    "run_plan",
    "investigate",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Planning is the reasoning-heavy step, so it gets the deeper thinking level by
# default. The fallback is used once if the first call times out.
DEFAULT_THINKING_LEVEL = "MEDIUM"
FALLBACK_THINKING_LEVEL = "LOW"

# A question should need a handful of steps, not twenty. A hard cap keeps a
# runaway plan from turning into a long, expensive series of tool calls.
MAX_STEPS = 5

# How many rows of a computed result to show in the report and in the evidence
# handed back to the model. Enough to see the shape, not the whole table.
MAX_PREVIEW_ROWS = 20

# Thinking tokens count towards this cap, so it must be comfortably larger than
# the JSON we want back. Gemini 3.6 Flash is a reasoning model: at MEDIUM
# thinking a tight cap leaves too little room for the answer and the output
# degrades (garbled characters inside string fields), so leave generous room.
PLAN_MAX_OUTPUT_TOKENS = 8192
INTERPRETATION_MAX_OUTPUT_TOKENS = 4096

# Distinct values shown per low-cardinality column in the dataset context. This
# is profiling information, not the dataset itself.
MAX_CATEGORICAL_VALUES = 12

# How many raw example values to show for a high-cardinality column.
MAX_EXAMPLES = 3


class InvestigationError(ValueError):
    """Base class for problems raised by this module."""


class PlanError(InvestigationError):
    """The model's plan failed validation and was rejected.

    Raised instead of silently dropping bad steps, so a malformed or
    out-of-schema plan is visible rather than quietly ignored.
    """


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    """One validated tool call: a known tool with clean, checked parameters.

    ``repairs`` records any value that had to be normalised before it could be
    accepted (see ``_snap_to_known``). It is normally empty; when it is not, the
    change is surfaced in the report rather than hidden.
    """

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    repairs: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """A one-line, human-readable description of the call."""
        rendered = json.dumps(self.params, ensure_ascii=False, sort_keys=True)
        label = f"`{self.tool}` {rendered}"
        return f"{label} — {self.purpose}" if self.purpose else label


@dataclass
class InvestigationPlan:
    """A validated plan, or an explicit decision that the question is unsupported."""

    status: str  # "plan" or "unsupported"
    steps: list[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    unsupported_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return self.status == "plan"

    @property
    def repairs(self) -> list[str]:
        """Every normalisation applied to this plan, across all steps."""
        return [note for step in self.steps for note in step.repairs]


@dataclass
class StepResult:
    """The outcome of executing one plan step.

    Exactly one of ``result`` and ``error`` is meaningful. A failed step is
    recorded rather than raised, so one bad tool call cannot sink a whole
    investigation that had other useful steps.
    """

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    result: pd.DataFrame | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None

    @property
    def n_rows(self) -> int:
        return 0 if self.result is None else int(len(self.result))


@dataclass
class Interpretation:
    """The model's narrative answer, produced strictly from computed evidence."""

    answer: str = ""
    interpretation: str = ""
    caveats: list[str] = field(default_factory=list)
    confidence: str = ""
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationReport:
    """The full result of one investigation.

    ``steps`` holds the computed evidence; ``interpretation`` holds the model's
    reading of it. They are deliberately separate fields so a caller (or a
    reader) can always tell which numbers were calculated and which words were
    generated.
    """

    question: str
    status: str  # "ok", "partial", "unsupported" or "error"
    plan: InvestigationPlan | None = None
    steps: list[StepResult] = field(default_factory=list)
    interpretation: Interpretation = field(default_factory=Interpretation)
    unverified_numbers: list[str] = field(default_factory=list)
    error: str | None = None
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def findings(self) -> list[StepResult]:
        """Only the steps that produced a computed result."""
        return [step for step in self.steps if step.ok]

    @property
    def failures(self) -> list[StepResult]:
        """Only the steps that failed."""
        return [step for step in self.steps if not step.ok]

    def to_markdown(self) -> str:
        """Render the report, keeping evidence and interpretation clearly apart."""
        lines: list[str] = [f"# Investigation: {self.question}", ""]
        lines.append(f"**Status:** `{self.status}`")
        if self.model:
            lines.append(f"**Model:** `{self.model}`")
        lines.append("")

        if self.error:
            lines.append(f"> {self.error}")
            lines.append("")

        if self.plan is not None:
            lines.append("## Plan (AI-generated, then validated)")
            if self.plan.reasoning:
                lines.append(self.plan.reasoning)
            if self.plan.is_supported:
                for index, step in enumerate(self.plan.steps, start=1):
                    lines.append(f"{index}. {step.describe()}")
            else:
                lines.append(
                    f"_Marked unsupported._ {self.plan.unsupported_reason}"
                )
            if self.plan.repairs:
                lines.append("")
                lines.append(
                    "**Normalised parameters** (the model's output contained "
                    "trailing text that was trimmed to a known value):"
                )
                for note in self.plan.repairs:
                    lines.append(f"- {note}")
            lines.append("")

        if self.steps:
            lines.append("## Findings (computed in Python)")
            for index, step in enumerate(self.steps, start=1):
                lines.append(f"### Step {index} — `{step.tool}`")
                if step.purpose:
                    lines.append(f"_{step.purpose}_")
                lines.append("")
                lines.append(
                    "Parameters: `"
                    + json.dumps(step.params, ensure_ascii=False, sort_keys=True)
                    + "`"
                )
                lines.append("")
                if step.ok:
                    lines.append(format_result_table(step.result))
                else:
                    lines.append(f"**Tool failed:** {step.error}")
                lines.append("")

        if self.interpretation.answer or self.interpretation.interpretation:
            lines.append("## Answer (AI interpretation — not independently verified)")
            if self.interpretation.answer:
                lines.append(self.interpretation.answer)
            if self.interpretation.interpretation:
                lines.append("")
                lines.append(self.interpretation.interpretation)
            if self.interpretation.confidence:
                lines.append("")
                lines.append(f"**Confidence:** {self.interpretation.confidence}")
            if self.interpretation.caveats:
                lines.append("")
                lines.append("### Caveats")
                for caveat in self.interpretation.caveats:
                    lines.append(f"- {caveat}")
            lines.append("")

        if self.unverified_numbers:
            lines.append("### Numbers not found in the evidence")
            lines.append(
                "These numbers appear in the AI interpretation but could not be "
                "matched to a computed result. Treat them with suspicion."
            )
            for number in self.unverified_numbers:
                lines.append(f"- `{number}`")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Context builders: what the model is allowed to see
# ---------------------------------------------------------------------------
def build_tool_catalogue() -> list[dict[str, Any]]:
    """Return the tool catalogue in a JSON-serialisable form.

    Delegates to ``src.tools.list_tools`` so the registry stays the single
    source of truth: adding a tool to the registry advertises it here
    automatically.
    """
    return list_tools()


def _is_date_like(series: pd.Series) -> bool:
    """Best-effort check for a text column that actually holds dates.

    CSVs deliver dates as strings, and the ``trend_over_time`` tool needs one.
    Telling the model which column is date-like is what lets it answer
    time-based questions without guessing.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return False

    sample = series.dropna().astype(str).head(50)
    if len(sample) < 2 or sample.nunique() < 2:
        return False
    # Plain numbers parse as dates in pandas (they are treated as epochs), which
    # would mislabel an id column. Skip anything that is purely numeric text.
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return False
    return bool(parsed.notna().all())


def _describe_column(name: str, series: pd.Series, stats: pd.Series | None) -> str:
    """One line of the dataset context for a single column."""
    total = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 1) if total else 0.0
    unique = int(series.nunique(dropna=True))

    if pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
    elif _is_date_like(series):
        kind = "text (date-like)"
    else:
        kind = "text"

    # A column is treated as categorical (and so gets its distinct values
    # listed) only when its values actually repeat. A column where every value
    # is unique is an identifier, not a category, and listing it would mean
    # sending a whole column.
    is_categorical = unique <= MAX_CATEGORICAL_VALUES and unique < total

    extras: list[str] = []
    if stats is not None and kind == "numeric":
        extras.append(
            "min {min:g}, mean {mean:g}, max {max:g}".format(
                min=float(stats["min"]),
                mean=float(stats["mean"]),
                max=float(stats["max"]),
            )
        )
    elif kind == "text (date-like)":
        examples = _sample_values(series, MAX_EXAMPLES)
        if examples:
            extras.append("examples: " + ", ".join(examples))
    elif kind == "text" and is_categorical:
        values = _sample_values(series, MAX_CATEGORICAL_VALUES)
        if values:
            extras.append("values: " + ", ".join(values))
    else:
        examples = _sample_values(series, MAX_EXAMPLES)
        if examples:
            extras.append("examples: " + ", ".join(examples))

    line = (
        f"- {name} | {kind} | missing {missing_pct}% "
        f"({missing} of {total}) | {unique} unique"
    )
    if extras:
        line += " | " + "; ".join(extras)
    return line


def _sample_values(series: pd.Series, limit: int) -> list[str]:
    """Distinct, non-null values rendered as short strings."""
    values: list[str] = []
    for value in series.dropna().unique():
        text = str(value)
        if len(text) > 40:
            text = text[:37] + "..."
        values.append(text)
        if len(values) >= limit:
            break
    return values


def build_dataset_context(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> str:
    """Describe the dataset's schema and profile as plain text.

    This is everything the model gets about the data: column names, types,
    missingness, uniqueness, descriptive statistics, distinct category values
    and a few examples. It never contains whole columns, and it is capped so it
    stays a fixed size no matter how large the dataset is.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if profile is None:
        profile = profile_dataset(df)

    lines = ["DATASET", f"- {profile['n_rows']} rows, {profile['n_columns']} columns"]
    lines.append(
        f"- {profile['total_missing']} missing cells "
        f"({profile['missing_pct']}% of all cells); "
        f"{profile['duplicate_rows']} duplicate rows"
    )
    if profile.get("warnings"):
        lines.append("- Data-quality warnings:")
        for warning in profile["warnings"]:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append("COLUMNS (name | type | missing | unique | extra)")

    numeric = numeric_summary(df)
    for name in df.columns:
        stats = None
        if name in numeric.index:
            stats = numeric.loc[name]
        lines.append(_describe_column(str(name), df[name], stats))

    lines.append("")
    lines.append(
        "Only the columns listed above exist. Parameters that name a column "
        "must use these exact names."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan schema and prompt
# ---------------------------------------------------------------------------
# Every parameter any tool accepts, so the model can express a plan in one
# object. Values that do not belong to the chosen tool are rejected by
# ``validate_plan`` afterwards.
_PARAM_PROPERTIES: dict[str, Any] = {
    "group_column": {"type": "string", "description": "Column whose categories define the groups."},
    "value_column": {"type": "string", "description": "Numeric column to summarise."},
    "metric": {"type": "string", "enum": ["mean", "median", "sum"]},
    "columns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Numeric columns to correlate.",
    },
    "method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]},
    "date_column": {"type": "string", "description": "Column holding dates."},
    "freq": {"type": "string", "description": "Pandas period alias, e.g. D, W, ME, QE, YE."},
    "column": {"type": "string", "description": "Single column to inspect."},
    "multiplier": {"type": "number", "description": "IQR sensitivity; 1.5 is standard."},
    "top_n": {"type": "integer", "description": "Maximum rows to return."},
    "target_column": {"type": "string", "description": "Column to check for missing values."},
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["plan", "unsupported"],
            "description": (
                "Use 'plan' when the question can be answered with the tools. "
                "Use 'unsupported' when no combination of them can answer it."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences explaining the choice of tools.",
        },
        "unsupported_reason": {
            "type": "string",
            "description": (
                "When status is 'unsupported', explain what would be needed to "
                "answer the question. Otherwise leave empty."
            ),
        },
        "steps": {
            "type": "array",
            "description": "Ordered tool calls. Empty when status is 'unsupported'.",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": list(TOOL_REGISTRY)},
                    "purpose": {
                        "type": "string",
                        "description": "What this step establishes.",
                    },
                    "params": {"type": "object", "properties": _PARAM_PROPERTIES},
                },
                "required": ["tool", "params"],
            },
        },
    },
    "required": ["status", "reasoning", "steps"],
}

PLAN_SYSTEM_INSTRUCTION = """\
You are the planning component of Data Detective, a data-analysis agent.

You choose from a fixed catalogue of Python tools. You cannot run code, and you
must never invent a result. The tools compute everything; you only decide which
ones to call and with what parameters.

Rules:
- Use only tools from the catalogue, and only the parameters each tool declares.
- Only reference columns that appear in the dataset schema.
- Choose the smallest plan that answers the question.
- If no combination of the available tools can answer the question, return
  status "unsupported" with a short, specific reason. Do not guess, and do not
  propose tools that are not in the catalogue.
- Return JSON only.
"""


def build_plan_prompt(
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> str:
    """Assemble the planning prompt: question, tool catalogue and schema."""
    catalogue = json.dumps(build_tool_catalogue(), ensure_ascii=False, indent=2)
    context = build_dataset_context(df, profile)
    return (
        f"QUESTION\n{question.strip()}\n\n"
        f"AVAILABLE TOOLS\n{catalogue}\n\n"
        f"{context}\n\n"
        f"Plan the investigation using at most {MAX_STEPS} steps."
    )


# ---------------------------------------------------------------------------
# Plan validation: the security boundary
# ---------------------------------------------------------------------------
def _is_column_param(name: str) -> bool:
    """True for parameters that name a single column (e.g. ``value_column``)."""
    return name == "column" or name.endswith("_column")


def _require_column(df: pd.DataFrame, column: str, where: str) -> None:
    if column not in df.columns:
        available = ", ".join(str(name) for name in df.columns)
        raise PlanError(
            f"{where} refers to unknown column '{column}'. "
            f"Available columns: {available}."
        )


def _coerce_number(value: Any, *, integer: bool, where: str) -> float | int:
    """Accept a real JSON number and reject everything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(
            f"{where} must be {'an integer' if integer else 'a number'}; "
            f"got {type(value).__name__}."
        )
    if integer:
        if float(value).is_integer():
            return int(value)
        raise PlanError(f"{where} must be a whole number; got {value!r}.")
    return float(value)


def _snap_to_known(value: str, candidates: list[str]) -> tuple[str, bool]:
    """Snap a value back to a known token when a model leaks trailing text.

    Gemini's structured output occasionally appends its own reasoning text to
    the end of a string field, e.g. ``"revenue metric: mean"`` where the
    intended value was ``"revenue"``. When a known token is a prefix of the
    value and the remainder is clearly not part of it, the value is snapped
    back to that token.

    A value that does not start with any known token is returned unchanged, so
    a genuinely wrong value is still rejected downstream. The second element of
    the result is ``True`` only when a change was made, so the caller can record
    it as a visible repair rather than hiding it.
    """
    if value in candidates:
        return value, False
    matches = [candidate for candidate in candidates if candidate and value.startswith(candidate)]
    if not matches:
        return value, False
    best = max(matches, key=len)
    remainder = value[len(best):]
    # Accept when the remainder does not continue the token as a word
    # ("revenue metric..." -> boundary) or when only one known token could
    # possibly match ("revenuepopular..." still means "revenue").
    boundary = not remainder or not remainder[0].isalnum()
    if boundary or len(matches) == 1:
        return best, True
    return value, False


def _validate_param(
    df: pd.DataFrame,
    *,
    tool: str,
    index: int,
    name: str,
    value: Any,
    declared: dict[str, Any],
    repairs: list[str],
) -> Any:
    """Validate one parameter value against its declared schema.

    Values are normalised against known columns and enums first (recording any
    change in ``repairs``), then the result is checked strictly. Nothing that
    still fails validation reaches ``run_tool``.
    """
    expected = declared.get("type", "string")
    where = f"Step {index} ({tool}) parameter '{name}'"
    allowed = declared.get("enum")

    if expected == "string":
        if not isinstance(value, str) or not value.strip():
            raise PlanError(f"{where} must be a non-empty string.")
        value = value.strip()
        if allowed and value not in allowed:
            snapped, changed = _snap_to_known(value, list(allowed))
            if changed:
                repairs.append(
                    f"{where} was normalised from {value!r} to {snapped!r}."
                )
                value = snapped
            else:
                raise PlanError(
                    f"{where} must be one of {', '.join(allowed)}; got {value!r}."
                )
        if _is_column_param(name) and value not in df.columns:
            snapped, changed = _snap_to_known(value, [str(c) for c in df.columns])
            if changed:
                repairs.append(
                    f"{where} was normalised from {value!r} to {snapped!r}."
                )
                value = snapped
            else:
                _require_column(df, value, where)
        return value

    if expected == "array":
        if not isinstance(value, list) or not value:
            raise PlanError(f"{where} must be a non-empty list.")
        items_type = declared.get("items", "string")
        cleaned: list[Any] = []
        for item in value:
            if items_type != "string":
                raise PlanError(
                    f"{where} has an unsupported item type {items_type!r}."
                )
            if not isinstance(item, str) or not item.strip():
                raise PlanError(f"{where} must contain non-empty strings.")
            item = item.strip()
            if name == "columns" and item not in df.columns:
                snapped, changed = _snap_to_known(item, [str(c) for c in df.columns])
                if changed:
                    repairs.append(
                        f"{where} was normalised from {item!r} to {snapped!r}."
                    )
                    item = snapped
                else:
                    _require_column(df, item, where)
            cleaned.append(item)
        return cleaned

    if expected in ("number", "integer"):
        return _coerce_number(value, integer=(expected == "integer"), where=where)

    raise PlanError(f"{where} has an unsupported schema type {expected!r}.")


def _validate_step(raw_step: Any, df: pd.DataFrame, index: int) -> PlanStep:
    """Validate one step: known tool, known parameters, correct types."""
    if not isinstance(raw_step, dict):
        raise PlanError(f"Step {index} must be a JSON object.")

    tool = raw_step.get("tool")
    if not isinstance(tool, str) or tool not in TOOL_REGISTRY:
        allowed = ", ".join(TOOL_REGISTRY)
        raise PlanError(
            f"Step {index} uses unknown tool {tool!r}. Allowed tools: {allowed}."
        )

    declared = TOOL_REGISTRY[tool]["parameters"]
    raw_params = raw_step.get("params")
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise PlanError(f"Step {index} ({tool}) 'params' must be a JSON object.")

    cleaned: dict[str, Any] = {}
    repairs: list[str] = []
    for name, value in raw_params.items():
        if value is None:
            # An explicit null means "not provided"; treat it as absent rather
            # than failing, since models routinely fill unused keys this way.
            continue
        if name not in declared:
            raise PlanError(
                f"Step {index} ({tool}) received unknown parameter '{name}'. "
                f"Allowed: {', '.join(declared) or 'none'}."
            )
        cleaned[name] = _validate_param(
            df,
            tool=tool,
            index=index,
            name=name,
            value=value,
            declared=declared[name],
            repairs=repairs,
        )

    missing = [
        name
        for name, spec in declared.items()
        if spec.get("required") and name not in cleaned
    ]
    if missing:
        raise PlanError(
            f"Step {index} ({tool}) is missing required parameter(s): "
            f"{', '.join(missing)}."
        )

    purpose = raw_step.get("purpose")
    return PlanStep(
        tool=tool,
        params=cleaned,
        purpose=purpose.strip() if isinstance(purpose, str) else "",
        repairs=repairs,
    )


def validate_plan(raw_plan: Any, df: pd.DataFrame) -> InvestigationPlan:
    """Validate a decoded plan, returning a clean one or raising ``PlanError``.

    This is the boundary that makes executing the plan safe: nothing reaches
    ``run_tool`` unless it names a registered tool, uses only that tool's
    parameters, supplies the required ones with the right types, and references
    columns that actually exist.
    """
    if not isinstance(raw_plan, dict):
        raise PlanError(
            f"The plan must be a JSON object, got {type(raw_plan).__name__}."
        )

    status = raw_plan.get("status")
    if status not in ("plan", "unsupported"):
        raise PlanError(
            f"Plan 'status' must be 'plan' or 'unsupported'; got {status!r}."
        )

    reasoning = raw_plan.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    reason = raw_plan.get("unsupported_reason")
    reason = reason.strip() if isinstance(reason, str) else ""

    raw_steps = raw_plan.get("steps")
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_steps, list):
        raise PlanError("Plan 'steps' must be a list.")

    if status == "unsupported":
        return InvestigationPlan(
            status="unsupported",
            steps=[],
            reasoning=reasoning,
            unsupported_reason=(
                reason or "No available tool can answer this question."
            ),
            raw=raw_plan,
        )

    if not raw_steps:
        raise PlanError("A plan with status 'plan' must contain at least one step.")
    if len(raw_steps) > MAX_STEPS:
        raise PlanError(
            f"The plan has {len(raw_steps)} steps, more than the limit of "
            f"{MAX_STEPS}."
        )

    steps = [
        _validate_step(raw_step, df, index)
        for index, raw_step in enumerate(raw_steps, start=1)
    ]
    return InvestigationPlan(
        status="plan", steps=steps, reasoning=reasoning, raw=raw_plan
    )


# ---------------------------------------------------------------------------
# Planning and execution
# ---------------------------------------------------------------------------
def _generate_json(
    client: Any,
    prompt: str,
    *,
    schema: dict[str, Any],
    system_instruction: str,
    thinking_level: str,
    fallback_thinking_level: str | None,
    max_output_tokens: int,
) -> LLMResponse:
    """Call the model, retrying once at a lower thinking level on timeout.

    The deeper thinking level gives better tool choices but is slower; if the
    free tier is slow enough to time out, one retry at the fallback level is
    usually enough to get an answer.
    """
    try:
        return client.generate_json(
            prompt,
            schema=schema,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
        )
    except LLMTimeoutError:
        if (
            not fallback_thinking_level
            or str(fallback_thinking_level).upper() == str(thinking_level).upper()
        ):
            raise
        return client.generate_json(
            prompt,
            schema=schema,
            system_instruction=system_instruction,
            thinking_level=fallback_thinking_level,
            max_output_tokens=max_output_tokens,
        )


def plan_investigation(
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
    *,
    client: Any,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
    fallback_thinking_level: str | None = FALLBACK_THINKING_LEVEL,
) -> InvestigationPlan:
    """Ask the model for a plan and validate it.

    Raises:
        LLMError: The model call failed.
        PlanError: The returned plan failed validation.
    """
    prompt = build_plan_prompt(question, df, profile)
    response = _generate_json(
        client,
        prompt,
        schema=PLAN_SCHEMA,
        system_instruction=PLAN_SYSTEM_INSTRUCTION,
        thinking_level=thinking_level,
        fallback_thinking_level=fallback_thinking_level,
        max_output_tokens=PLAN_MAX_OUTPUT_TOKENS,
    )
    plan = validate_plan(response.json(), df)
    plan.raw.setdefault("_model", response.model)
    return plan


def run_plan(df: pd.DataFrame, plan: InvestigationPlan) -> list[StepResult]:
    """Execute every step through ``run_tool``, capturing failures per step.

    A tool that raises is recorded as a failed ``StepResult`` rather than
    propagated, so a single bad call does not discard the results of the others.
    """
    results: list[StepResult] = []
    for step in plan.steps:
        try:
            computed = run_tool(step.tool, df, **step.params)
            results.append(
                StepResult(
                    tool=step.tool,
                    params=dict(step.params),
                    purpose=step.purpose,
                    result=computed,
                    error=None,
                )
            )
        except ToolError as exc:
            results.append(
                StepResult(
                    tool=step.tool,
                    params=dict(step.params),
                    purpose=step.purpose,
                    result=None,
                    error=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the run
            results.append(
                StepResult(
                    tool=step.tool,
                    params=dict(step.params),
                    purpose=step.purpose,
                    result=None,
                    error=f"Unexpected error while running '{step.tool}': {exc}",
                )
            )
    return results


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------
def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # pragma: no cover - non-scalar cell
        pass
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("|", "\\|")


def format_result_table(df: pd.DataFrame | None, max_rows: int = MAX_PREVIEW_ROWS) -> str:
    """Render a computed result as a small Markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` so it needs no
    extra dependency, and capped so a large result never floods the report.
    """
    if df is None:
        return "_(no result)_"
    if len(df) == 0:
        return "_(no rows returned)_"

    head = df.head(max_rows)
    columns = [str(column) for column in head.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in head.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in head.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"_(showing the first {max_rows} of {len(df)} rows)_")
    return "\n".join(lines)


def build_evidence(steps: list[StepResult]) -> str:
    """Render the computed steps as the evidence block handed to the model."""
    blocks: list[str] = []
    for index, step in enumerate(steps, start=1):
        header = f"STEP {index} — {step.tool}"
        if step.purpose:
            header += f"\nPurpose: {step.purpose}"
        params = json.dumps(step.params, ensure_ascii=False, sort_keys=True)
        if step.ok:
            body = (
                f"Parameters: {params}\n"
                f"Result ({step.n_rows} rows):\n"
                f"{format_result_table(step.result)}"
            )
        else:
            body = f"Parameters: {params}\nResult: FAILED — {step.error}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Interpretation and the hallucination check
# ---------------------------------------------------------------------------
INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "A direct answer to the question, using only numbers that "
                "appear in the evidence."
            ),
        },
        "interpretation": {
            "type": "string",
            "description": (
                "What the evidence means, and how strongly it supports the "
                "answer. Separate what the data shows from what you infer."
            ),
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Limitations: missing data, small groups, weak effects, or "
                "reasons the answer might mislead."
            ),
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "interpretation", "confidence"],
}

INTERPRETATION_SYSTEM_INSTRUCTION = """\
You are the reporting component of Data Detective.

You are given evidence computed by Python tools. It is the only source of
numbers you may use.

Rules:
- Use only numbers and category names that appear in the evidence. Never
  calculate a new statistic, estimate, extrapolate or round differently.
- If the evidence does not answer the question, say so plainly instead of
  guessing.
- Distinguish what the data shows from what you infer.
- Correlation is not causation; never claim one thing caused another.
- Mention missing data, small groups or weak effects when they matter.
- Be concise and specific. Return JSON only.
"""


def build_interpretation_prompt(
    question: str,
    df: pd.DataFrame,
    steps: list[StepResult],
    profile: dict[str, Any] | None = None,
) -> str:
    """Assemble the reporting prompt: question, schema and computed evidence."""
    context = build_dataset_context(df, profile)
    evidence = build_evidence(steps)
    return (
        f"QUESTION\n{question.strip()}\n\n"
        f"{context}\n\n"
        "EVIDENCE (computed by Python — the only numbers you may use)\n"
        f"{evidence}\n\n"
        "Answer the question from the evidence above."
    )


def _numbers_in(text: str) -> list[str]:
    """Every numeric-looking token in a string, as written."""
    return re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")


def _normalise_number(token: str) -> float | None:
    cleaned = token.replace(",", "").rstrip(".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _matches_evidence(token: str, known: list[float]) -> bool:
    """True when a token is an evidence value, possibly re-expressed.

    A model that writes "912.01" for a computed "912.007" is rounding, not
    inventing. Accepting a value within half a unit of the token's last decimal
    place keeps the check useful without crying wolf over rounding. Signs are
    ignored because the surrounding sentence carries the direction ("fell"),
    and a factor of 100 either way is accepted so a percentage and its fraction
    ("0.028" for "2.8%") are treated as the same number.
    """
    value = _normalise_number(token)
    if value is None:
        return False
    tolerance = 0.5 * 10 ** (-_decimals(token))
    for scale in (1.0, 100.0, 0.01):
        scaled = value * scale
        limit = tolerance * scale
        if any(abs(abs(known_value) - abs(scaled)) <= limit for known_value in known):
            return True
    return False


def _is_significant(token: str) -> bool:
    """Only flag numbers that look like statistics, not counts of words.

    Plain small integers ("2 tools", "the 3 regions") are far too common to be
    useful signals, so the check focuses on numbers with decimals, thousands
    separators, or three or more digits.
    """
    cleaned = token.replace(",", "")
    return "." in cleaned or "," in token or len(cleaned.lstrip("-")) >= 3


def find_unverified_numbers(text: str, evidence: str) -> list[str]:
    """Return numbers in ``text`` that do not appear in ``evidence``.

    A best-effort guard against invented statistics: it flags numbers in the
    narrative that cannot be matched to a computed value, allowing for
    rounding. It is deliberately conservative (only "statistical-looking"
    numbers are considered) and it can still be fooled, so its output is
    surfaced as a caution, never as proof.
    """
    known = [
        value
        for token in _numbers_in(evidence)
        if (value := _normalise_number(token)) is not None
    ]
    flagged: list[str] = []
    for token in _numbers_in(text):
        if not _is_significant(token):
            continue
        if _matches_evidence(token, known):
            continue
        if token not in flagged:
            flagged.append(token)
    return flagged


def _interpretation_from(
    payload: Any, response: LLMResponse
) -> Interpretation:
    """Turn a validated interpretation payload into an ``Interpretation``."""
    if not isinstance(payload, dict):
        raise InvestigationError(
            "The interpretation response was not a JSON object."
        )
    caveats = payload.get("caveats")
    if not isinstance(caveats, list):
        caveats = []
    return Interpretation(
        answer=str(payload.get("answer", "")).strip(),
        interpretation=str(payload.get("interpretation", "")).strip(),
        caveats=[str(item).strip() for item in caveats if str(item).strip()],
        confidence=str(payload.get("confidence", "")).strip(),
        model=response.model,
        usage=dict(response.usage or {}),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _error_report(
    question: str,
    message: str,
    *,
    plan: InvestigationPlan | None = None,
    steps: list[StepResult] | None = None,
    status: str = "error",
) -> InvestigationReport:
    return InvestigationReport(
        question=question,
        status=status,
        plan=plan,
        steps=steps or [],
        error=message,
    )


def investigate(
    question: str,
    df: pd.DataFrame,
    *,
    profile: dict[str, Any] | None = None,
    client: Any | None = None,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
    fallback_thinking_level: str | None = FALLBACK_THINKING_LEVEL,
    interpret: bool = True,
) -> InvestigationReport:
    """Run a full investigation: plan, execute, then interpret the evidence.

    This is the graceful entry point. Expected failures (a bad question, an API
    error, a rejected plan, tools that all fail) come back as a report with a
    ``status`` and an ``error`` message rather than as an exception, so a UI can
    always render something honest.

    Args:
        question: The user's natural-language question.
        df: The dataset to investigate.
        profile: Optional precomputed profile; recomputed when omitted.
        client: An object exposing ``generate_json(...)``. Defaults to
            ``GeminiClient`` built from the environment.
        thinking_level: Thinking depth for the model calls.
        fallback_thinking_level: Retried once on timeout when it differs.
        interpret: When False, stop after executing the plan. Useful for
            scoring tool selection without paying for the reporting call.
    """
    question = question if isinstance(question, str) else ""
    if not question.strip():
        return _error_report(question, "Please provide a non-empty question.")
    question = question.strip()

    if not isinstance(df, pd.DataFrame):
        return _error_report(
            question, "The dataset must be a pandas DataFrame."
        )
    if df.shape[1] == 0 or len(df) == 0:
        return _error_report(
            question, "The dataset has no rows or no columns to analyse."
        )

    if profile is None:
        profile = profile_dataset(df)

    if client is None:
        try:
            client = GeminiClient()
        except LLMError as exc:
            return _error_report(question, str(exc))

    try:
        plan = plan_investigation(
            question,
            df,
            profile,
            client=client,
            thinking_level=thinking_level,
            fallback_thinking_level=fallback_thinking_level,
        )
    except LLMError as exc:
        return _error_report(question, f"The planning request failed: {exc}")
    except PlanError as exc:
        return _error_report(
            question, f"The model's plan was rejected: {exc}"
        )

    model = str(plan.raw.get("_model", "")) if plan.raw else ""

    if not plan.is_supported:
        return InvestigationReport(
            question=question,
            status="unsupported",
            plan=plan,
            model=model,
        )

    steps = run_plan(df, plan)

    if not any(step.ok for step in steps):
        detail = "; ".join(
            f"{step.tool}: {step.error}" for step in steps if step.error
        )
        return _error_report(
            question,
            "Every tool call failed"
            + (f" ({detail})." if detail else "."),
            plan=plan,
            steps=steps,
        )

    if not interpret:
        return InvestigationReport(
            question=question,
            status="ok",
            plan=plan,
            steps=steps,
            model=model,
        )

    try:
        prompt = build_interpretation_prompt(question, df, steps, profile)
        response = _generate_json(
            client,
            prompt,
            schema=INTERPRETATION_SCHEMA,
            system_instruction=INTERPRETATION_SYSTEM_INSTRUCTION,
            thinking_level=thinking_level,
            fallback_thinking_level=fallback_thinking_level,
            max_output_tokens=INTERPRETATION_MAX_OUTPUT_TOKENS,
        )
        interpretation = _interpretation_from(response.json(), response)
    except LLMError as exc:
        report = InvestigationReport(
            question=question,
            status="partial",
            plan=plan,
            steps=steps,
            model=model,
            error=(
                "The computed findings are shown below, but the AI "
                f"interpretation failed: {exc}"
            ),
        )
        return report
    except InvestigationError as exc:
        return InvestigationReport(
            question=question,
            status="partial",
            plan=plan,
            steps=steps,
            model=model,
            error=f"The AI interpretation was unusable: {exc}",
        )

    narrative = " ".join(
        [interpretation.answer, interpretation.interpretation]
        + interpretation.caveats
    )
    unverified = find_unverified_numbers(narrative, build_evidence(steps))

    return InvestigationReport(
        question=question,
        status="ok",
        plan=plan,
        steps=steps,
        interpretation=interpretation,
        unverified_numbers=unverified,
        model=model or interpretation.model,
        usage=interpretation.usage,
    )
