"""Presentation helpers for the interface layer.

These functions turn an :class:`~src.investigator.InvestigationReport` (and a
dataset) into plain data that any front end can render. They deliberately
contain **no Streamlit import**: the UI is a thin renderer over this module, so
everything here is unit-tested offline.

Public functions:
    status_banner(report)          -> banner level/title/message for the report
    plan_lines(report)             -> the validated plan, one line per step
    finding_sections(report)       -> computed evidence, ready to render as tables
    interpretation_section(report) -> the AI narrative, kept separate from evidence
    unverified_warning(report)     -> caution text when the narrative invented numbers
    redaction_candidates(df)       -> columns that look like identifiers/PII
    example_questions(df)          -> starter questions derived from the real schema
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.investigator import InvestigationReport, format_result_table

__all__ = [
    "STATUS_STYLES",
    "status_banner",
    "plan_lines",
    "finding_sections",
    "interpretation_section",
    "unverified_warning",
    "redaction_candidates",
    "example_questions",
    "dataset_label",
]

# Map an investigation status onto a UI banner level. Streamlit understands
# "success", "warning", "error" and "info".
STATUS_STYLES: dict[str, str] = {
    "ok": "success",
    "partial": "warning",
    "unsupported": "info",
    "error": "error",
}

_STATUS_TITLES: dict[str, str] = {
    "ok": "Investigation complete",
    "partial": "Findings computed — the AI interpretation failed",
    "unsupported": "This question cannot be answered from this dataset",
    "error": "The investigation could not be completed",
}

# Column-name fragments that usually mean "identifier or personal data". Used
# only to *suggest* columns for redaction; the user always makes the choice.
PII_HINTS: tuple[str, ...] = (
    "id",
    "email",
    "e-mail",
    "phone",
    "mobile",
    "name",
    "address",
    "postcode",
    "zip",
    "ssn",
    "passport",
    "dob",
    "birth",
    "account",
    "iban",
    "card",
    "customer",
)

# A column is "high cardinality" (and therefore identifier-like) above this
# share of distinct values.
HIGH_CARDINALITY_SHARE = 0.9


def dataset_label(name: str, df: pd.DataFrame) -> str:
    """A short, human-readable label for the loaded dataset."""
    label = (name or "dataset").strip() or "dataset"
    return f"{label} ({len(df):,} rows × {df.shape[1]:,} columns)"


def status_banner(report: InvestigationReport) -> dict[str, str]:
    """Return ``{"level", "title", "message"}`` describing the run's outcome."""
    level = STATUS_STYLES.get(report.status, "info")
    title = _STATUS_TITLES.get(report.status, "Investigation finished")

    if report.error:
        message = report.error
    elif report.status == "unsupported":
        message = (
            report.plan.unsupported_reason.strip()
            if report.plan and report.plan.unsupported_reason.strip()
            else "No available tool can answer this question from this dataset."
        )
    elif report.status == "ok":
        message = (
            "The findings below were computed in Python. The answer is the "
            "model's reading of those findings, not a source of new numbers."
        )
    else:  # pragma: no cover - every status above is handled
        message = ""
    return {"level": level, "title": title, "message": message}


def plan_lines(report: InvestigationReport) -> list[str]:
    """Describe the validated plan as one line per step.

    Returns an empty list when no plan was produced. An unsupported plan
    produces a single explanatory line.
    """
    plan = report.plan
    if plan is None:
        return []
    if not plan.is_supported:
        reason = plan.unsupported_reason.strip() or "no tool fits this question"
        return [f"Marked unsupported: {reason}"]
    return [step.describe() for step in plan.steps]


def finding_sections(report: InvestigationReport) -> list[dict[str, Any]]:
    """Return the computed evidence as renderable sections.

    Each section is a dict with the keys ``title``, ``purpose``, ``params``,
    ``frame`` (a DataFrame or ``None``), ``table`` (Markdown) and ``error``.
    """
    sections: list[dict[str, Any]] = []
    for index, step in enumerate(report.steps, start=1):
        sections.append(
            {
                "title": f"Step {index} — {step.tool}",
                "tool": step.tool,
                "purpose": step.purpose,
                "params": step.params,
                "frame": step.result if step.ok else None,
                "table": format_result_table(step.result) if step.ok else "",
                "error": step.error,
            }
        )
    return sections


def interpretation_section(report: InvestigationReport) -> dict[str, Any] | None:
    """Return the AI narrative, or ``None`` when there is nothing to show."""
    interpretation = report.interpretation
    if not (interpretation.answer or interpretation.interpretation):
        return None
    return {
        "answer": interpretation.answer,
        "interpretation": interpretation.interpretation,
        "confidence": interpretation.confidence,
        "caveats": list(interpretation.caveats),
        "model": interpretation.model or report.model,
    }


def unverified_warning(report: InvestigationReport) -> str | None:
    """Caution text when the narrative contains numbers no tool computed."""
    if not report.unverified_numbers:
        return None
    listed = ", ".join(f"`{number}`" for number in report.unverified_numbers)
    return (
        "These numbers appear in the AI answer but could not be matched to a "
        f"computed result: {listed}. Treat them with suspicion."
    )


def redaction_candidates(df: pd.DataFrame, limit: int = 12) -> list[str]:
    """Suggest columns that look like identifiers or personal data.

    A column is suggested when its name contains a known identifier fragment or
    when it is a high-cardinality non-numeric column (every value distinct). The
    list is a suggestion only: redaction is opt-in and chosen by the user.
    """
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return []

    total = len(df)
    suggestions: list[str] = []
    for name in df.columns:
        series = df[name]
        lowered = str(name).lower()
        looks_like_id = any(hint in lowered for hint in PII_HINTS)
        unique = int(series.nunique(dropna=True))
        high_cardinality = (
            total > 0
            and not pd.api.types.is_numeric_dtype(series)
            and unique >= max(2, int(total * HIGH_CARDINALITY_SHARE))
        )
        if looks_like_id or high_cardinality:
            suggestions.append(str(name))
    return suggestions[:limit]


def _looks_like_dates(series: pd.Series) -> bool:
    """Best-effort check for a text column that actually holds dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.8)


def example_questions(df: pd.DataFrame, limit: int = 4) -> list[str]:
    """Starter questions built from the dataset's real columns.

    Nothing here is specific to a particular dataset: the suggestions are
    derived from column names and types, so they generalise to any CSV.
    """
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return []

    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    others = [str(c) for c in df.columns if str(c) not in numeric]
    dates = [c for c in others if _looks_like_dates(df[c])]
    redacted = set(redaction_candidates(df))
    categorical = [
        c
        for c in others
        if c not in dates
        and c not in redacted
        and 1 < int(df[c].nunique(dropna=True)) <= max(2, len(df) // 3)
    ]

    questions: list[str] = []
    if numeric and categorical:
        questions.append(f"Compare {numeric[0]} across {categorical[0]}")
    if dates and numeric:
        questions.append(f"How has {numeric[0]} changed over time?")
    if len(numeric) >= 2:
        questions.append(f"Is {numeric[0]} correlated with {numeric[1]}?")
    if numeric:
        questions.append(f"Which rows are outliers in {numeric[0]}?")
    if categorical:
        questions.append(f"How many rows are in each {categorical[0]}?")
    if numeric and categorical:
        questions.append(
            f"Which {categorical[0]} has the most missing {numeric[0]}?"
        )
    return questions[:limit]
