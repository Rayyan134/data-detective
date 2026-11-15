"""Charts built from computed tool results.

Every chart here is derived from a DataFrame a tool already produced — never
from the model. A function returns ``None`` when a result cannot be charted
cleanly, so the UI simply skips the chart instead of crashing.

Public functions:
    chart_for(tool, frame, params=None, *, title=None) -> altair.Chart | None
    chart_group_comparison(frame, ...) -> altair.Chart | None
    chart_value_counts(frame, ...)    -> altair.Chart | None
    chart_trend(frame, ...)          -> altair.Chart | None
    chart_missing(frame, ...)        -> altair.Chart | None
    chart_correlation(frame, ...)    -> altair.Chart | None
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

__all__ = [
    "chart_for",
    "chart_group_comparison",
    "chart_value_counts",
    "chart_trend",
    "chart_missing",
    "chart_correlation",
]

# Keep charts compact so they fit the report column.
CHART_HEIGHT = 260
CHART_WIDTH = 520


def _safe(callable_, *args, **kwargs):
    """Run a chart builder; return None on any failure."""
    try:
        return callable_(*args, **kwargs)
    except Exception:  # noqa: BLE001 - a bad frame must not break the UI
        return None


def _base(title: str | None) -> dict[str, Any]:
    return {"title": title} if title else {}


def chart_group_comparison(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    params = params or {}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    group_col = params.get("group_column") or frame.columns[0]
    metric = params.get("metric") or "mean"
    if metric not in frame.columns:
        metric = "mean" if "mean" in frame.columns else frame.columns[-1]
    if group_col not in frame.columns or metric not in frame.columns:
        return None
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
            y=alt.Y(f"{metric}:Q", title=metric),
            tooltip=[group_col, metric, "count"] if "count" in frame.columns else [group_col, metric],
        )
        .properties(height=CHART_HEIGHT, width=CHART_WIDTH, **_base(title))
    )
    return chart


def chart_value_counts(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    params = params or {}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    value_col = params.get("column") or frame.columns[0]
    if value_col not in frame.columns or "count" not in frame.columns:
        return None
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:N", sort="-y", title=value_col),
            y=alt.Y("count:Q", title="count"),
            tooltip=[value_col, "count", "pct"] if "pct" in frame.columns else [value_col, "count"],
        )
        .properties(height=CHART_HEIGHT, width=CHART_WIDTH, **_base(title))
    )
    return chart


def chart_trend(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "period" not in frame.columns or "total" not in frame.columns:
        return None
    line = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:N", title="period", sort=None),
            y=alt.Y("total:Q", title="total"),
            tooltip=["period", "total", "orders", "pct_change"]
            if all(c in frame.columns for c in ("orders", "pct_change"))
            else ["period", "total"],
        )
        .properties(height=CHART_HEIGHT, width=CHART_WIDTH, **_base(title))
    )
    return line


def chart_missing(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    params = params or {}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    group_col = params.get("group_column") or frame.columns[0]
    if group_col not in frame.columns or "missing_pct" not in frame.columns:
        return None
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
            y=alt.Y("missing_pct:Q", title="missing %"),
            tooltip=[group_col, "missing", "rows", "missing_pct"]
            if all(c in frame.columns for c in ("missing", "rows"))
            else [group_col, "missing_pct"],
        )
        .properties(height=CHART_HEIGHT, width=CHART_WIDTH, **_base(title))
    )
    return chart


def chart_correlation(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[0] < 2:
        return None
    melted = frame.reset_index().melt(id_vars=frame.index.name or "index", var_name="variable", value_name="value")
    x_col = frame.index.name or "index"
    if x_col not in melted.columns:
        melted = melted.rename(columns={"index": x_col})
    chart = (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_col}:N", title=x_col),
            y=alt.Y("variable:N", title="variable"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=[x_col, "variable", "value"],
        )
        .properties(height=CHART_HEIGHT, width=CHART_WIDTH, **_base(title))
    )
    return chart


_BUILDERS = {
    "group_comparison": chart_group_comparison,
    "value_counts_table": chart_value_counts,
    "trend_over_time": chart_trend,
    "missing_by_group": chart_missing,
    "correlation": chart_correlation,
}


def chart_for(
    tool: str,
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    title: str | None = None,
) -> alt.Chart | None:
    """Return a chart for a tool's result frame, or None if not chartable."""
    builder = _BUILDERS.get(tool)
    if builder is None:
        return None
    return _safe(builder, frame, params, title=title)
