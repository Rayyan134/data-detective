"""Offline tests for src.visualizer (charts from computed results)."""

from __future__ import annotations

import pandas as pd
import pytest

import altair as alt

from src.visualizer import (
    chart_for,
    chart_correlation,
    chart_group_comparison,
    chart_missing,
    chart_trend,
    chart_value_counts,
)


def test_group_comparison_chart():
    frame = pd.DataFrame({"region": ["East", "West"], "count": [5, 3], "mean": [10.0, 20.0]})
    chart = chart_group_comparison(frame, {"group_column": "region", "metric": "mean"})
    assert isinstance(chart, alt.Chart)


def test_value_counts_chart():
    frame = pd.DataFrame({"channel": ["Online", "Retail"], "count": [94, 52], "pct": [64.4, 35.6]})
    chart = chart_value_counts(frame, {"column": "channel"})
    assert isinstance(chart, alt.Chart)


def test_trend_chart():
    frame = pd.DataFrame({"period": ["2025-01", "2025-02"], "total": [100.0, 120.0], "orders": [5, 6]})
    chart = chart_trend(frame)
    assert isinstance(chart, alt.Chart)


def test_missing_chart():
    frame = pd.DataFrame({"region": ["East", "West"], "missing": [1, 2], "rows": [10, 10], "missing_pct": [10.0, 20.0]})
    chart = chart_missing(frame, {"group_column": "region"})
    assert isinstance(chart, alt.Chart)


def test_correlation_chart():
    frame = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])
    chart = chart_correlation(frame)
    assert isinstance(chart, alt.Chart)


def test_chart_for_dispatch():
    frame = pd.DataFrame({"region": ["East"], "count": [5], "mean": [10.0]})
    assert chart_for("group_comparison", frame, {"group_column": "region", "metric": "mean"}) is not None
    assert chart_for("value_counts_table", pd.DataFrame({"x": [1], "count": [1]}), {"column": "x"}) is not None
    assert chart_for("find_outliers", pd.DataFrame({"a": [1]})) is None
    assert chart_for("unknown_tool", pd.DataFrame({"a": [1]})) is None


def test_charts_handle_empty_and_bad_frames():
    empty = pd.DataFrame()
    assert chart_for("group_comparison", empty) is None
    assert chart_for("value_counts_table", empty) is None
    assert chart_for("trend_over_time", empty) is None
    assert chart_for("missing_by_group", empty) is None
    assert chart_for("correlation", empty) is None
    # Non-DataFrame input must not raise.
    assert chart_for("group_comparison", None) is None
