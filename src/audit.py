"""Safe, structured logging of investigations.

Writes one JSON object per line (JSON Lines). The log records *metadata about*
an investigation - the question, the dataset's shape, the tools that ran, token
usage and the outcome - but **never the dataset itself**: no cell values, no
evidence tables, and never the API key.

That boundary is what makes the log safe to keep: a reviewer can see how the
agent behaved without the log becoming a copy of the user's data.

Public functions:
    build_log_record(**kwargs)  -> the dict that would be written (pure)
    log_investigation(path, **kwargs) -> build + append, returns the record
    read_log(path)              -> parse a log file back into dicts (for tests/tools)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.investigator import InvestigationReport

__all__ = [
    "MAX_QUESTION_CHARS",
    "MAX_ERROR_CHARS",
    "DEFAULT_LOG_PATH",
    "build_log_record",
    "log_investigation",
    "read_log",
    "token_summary",
]

# Keep free-text fields short so a log line stays a log line.
MAX_QUESTION_CHARS = 200
MAX_ERROR_CHARS = 200

# Logs live beside the app but are git-ignored (*.log is ignored; this file has
# a .jsonl extension, so logs/ is ignored explicitly in .gitignore).
DEFAULT_LOG_PATH = Path("logs") / "investigations.jsonl"


def token_summary(usage: dict[str, Any] | None) -> dict[str, int]:
    """Reduce a Gemini ``usageMetadata`` dict to four token counts."""
    usage = usage or {}
    return {
        "prompt": int(usage.get("promptTokenCount", 0) or 0),
        "output": int(usage.get("candidatesTokenCount", 0) or 0),
        "thinking": int(usage.get("thoughtsTokenCount", 0) or 0),
        "total": int(usage.get("totalTokenCount", 0) or 0),
    }


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def build_log_record(
    *,
    question: str,
    report: InvestigationReport,
    dataset_label: str = "",
    n_rows: int = 0,
    n_columns: int = 0,
    redact_columns: Iterable[str] | None = None,
    latency_ms: int | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build the log record for one investigation.

    Pure: no file is touched and no clock is read unless ``timestamp`` is
    omitted. Only column *names* and parameter values that the validator already
    accepted (column names, enums, numbers) are recorded - never cell values.
    """
    moment = timestamp or datetime.now(timezone.utc)
    plan = report.plan
    return {
        "timestamp": moment.isoformat(timespec="seconds"),
        "question": _clip(question, MAX_QUESTION_CHARS),
        "dataset": {
            "label": str(dataset_label),
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
        },
        "redacted_columns": sorted({str(c) for c in (redact_columns or [])}),
        "status": report.status,
        "model": report.model,
        "plan_supported": bool(plan.is_supported) if plan is not None else None,
        "plan_repairs": len(plan.repairs) if plan is not None else 0,
        "tools": [step.tool for step in report.steps],
        "parameters": {step.tool: dict(step.params) for step in report.steps},
        "steps_ok": len(report.findings),
        "steps_failed": len(report.failures),
        "unverified_numbers": len(report.unverified_numbers),
        "tokens": token_summary(report.usage),
        "latency_ms": latency_ms,
        "error": _clip(report.error or "", MAX_ERROR_CHARS),
    }


def log_investigation(
    path: str | Path = DEFAULT_LOG_PATH,
    *,
    question: str,
    report: InvestigationReport,
    dataset_label: str = "",
    n_rows: int = 0,
    n_columns: int = 0,
    redact_columns: Iterable[str] | None = None,
    latency_ms: int | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Append one investigation record to ``path`` and return it.

    Creates the parent directory on first use. Logging never raises on an I/O
    problem: a failed log must not take down a finished investigation, so the
    error is swallowed and the record is still returned.
    """
    record = build_log_record(
        question=question,
        report=report,
        dataset_label=dataset_label,
        n_rows=n_rows,
        n_columns=n_columns,
        redact_columns=redact_columns,
        latency_ms=latency_ms,
        timestamp=timestamp,
    )
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:  # pragma: no cover - disk problems must not crash the app
        pass
    return record


def read_log(path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """Read a JSONL log back into a list of records (blank lines are skipped)."""
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
