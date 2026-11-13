"""Tests for src.tools.

Each test builds a tiny dataset with known answers, so the expected numbers can
be verified by hand. Run with:

    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.tools import (
    TOOL_REGISTRY,
    ToolError,
    correlation,
    find_outliers,
    group_comparison,
    list_tools,
    missing_by_group,
    run_tool,
    trend_over_time,
    value_counts_table,
)


@pytest.fixture
def sales() -> pd.DataFrame:
    """A small dataset with hand-checkable values."""
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "West"],
            "product": ["A", "B", "A", "B", "A"],
            "channel": ["Online", "Retail", "Online", "Partner", "Retail"],
            "units": [10, 20, 30, 40, 50],
            "revenue": [100.0, 250.0, 300.0, 420.0, 500.0],
            "satisfaction": [5.0, None, 3.0, 4.0, None],
            "order_date": [
                "2025-01-05",
                "2025-01-20",
                "2025-02-03",
                "2025-02-18",
                "2025-03-01",
            ],
        }
    )


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------
def test_group_comparison_sorts_by_mean_descending(sales):
    result = group_comparison(sales, "region", "revenue", metric="mean")

    assert list(result["region"]) == ["West", "South", "North"]
    assert result.loc[result["region"] == "North", "count"].iloc[0] == 2
    assert result.loc[result["region"] == "West", "mean"].iloc[0] == 500.0


def test_group_comparison_can_total_by_sum(sales):
    result = group_comparison(sales, "product", "revenue", metric="sum")

    assert list(result["product"]) == ["A", "B"]
    assert result.loc[result["product"] == "A", "sum"].iloc[0] == 900.0


def test_group_comparison_rejects_unknown_column(sales):
    with pytest.raises(ToolError, match="not found"):
        group_comparison(sales, "country", "revenue")


def test_group_comparison_rejects_non_numeric_value(sales):
    with pytest.raises(ToolError, match="not numeric"):
        group_comparison(sales, "region", "product")


def test_group_comparison_rejects_bad_metric(sales):
    with pytest.raises(ToolError, match="metric must be"):
        group_comparison(sales, "region", "revenue", metric="mode")


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------
def test_correlation_matrix_is_square_with_unit_diagonal(sales):
    # No columns given, so every numeric column is included: units, revenue
    # and satisfaction.
    result = correlation(sales)

    assert list(result.columns) == ["units", "revenue", "satisfaction"]
    assert result.shape == (3, 3)
    assert result.loc["units", "units"] == 1.0
    # Strongly positive but not exactly 1: the points rise together yet not in
    # a perfectly straight line, and Pearson measures linearity.
    assert result.loc["units", "revenue"] == pytest.approx(0.99, abs=0.01)


def test_correlation_accepts_a_column_subset(sales):
    result = correlation(sales, columns=["revenue", "units"])

    assert result.shape == (2, 2)


def test_correlation_needs_two_numeric_columns(sales):
    with pytest.raises(ToolError, match="at least two"):
        correlation(sales, columns=["revenue"])


def test_correlation_rejects_non_numeric_column(sales):
    with pytest.raises(ToolError, match="not numeric"):
        correlation(sales, columns=["revenue", "product"])


# ---------------------------------------------------------------------------
# trend_over_time
# ---------------------------------------------------------------------------
def test_trend_over_time_totals_by_month(sales):
    result = trend_over_time(sales, "order_date", "revenue", freq="ME")

    assert list(result["total"]) == [350.0, 720.0, 500.0]
    assert list(result["orders"]) == [2, 2, 1]
    assert result["change"].tolist()[1] == 370.0
    assert result["pct_change"].tolist()[1] == 105.7
    assert result["pct_change"].tolist()[2] == -30.6


def test_trend_over_time_rejects_invalid_dates(sales):
    broken = sales.copy()
    broken.loc[0, "order_date"] = "not-a-date"

    with pytest.raises(ToolError, match="could not be parsed"):
        trend_over_time(broken, "order_date", "revenue")


def test_trend_over_time_rejects_invalid_frequency(sales):
    with pytest.raises(ToolError, match="frequency"):
        trend_over_time(sales, "order_date", "revenue", freq="banana")


# ---------------------------------------------------------------------------
# find_outliers
# ---------------------------------------------------------------------------
def test_find_outliers_flags_the_high_value():
    df = pd.DataFrame({"value": [10.0, 11.0, 12.0, 13.0, 100.0]})

    result = find_outliers(df, "value")

    assert len(result) == 1
    assert result["value"].iloc[0] == 100.0
    assert result["iqr_lower_bound"].iloc[0] == 8.0
    assert result["iqr_upper_bound"].iloc[0] == 16.0


def test_find_outliers_returns_empty_frame_when_none_found(sales):
    result = find_outliers(sales, "revenue")

    assert result.empty
    assert "iqr_upper_bound" in result.columns


def test_find_outliers_rejects_non_positive_multiplier(sales):
    with pytest.raises(ToolError, match="greater than zero"):
        find_outliers(sales, "revenue", multiplier=0)


# ---------------------------------------------------------------------------
# value_counts_table
# ---------------------------------------------------------------------------
def test_value_counts_table_counts_and_percentages(sales):
    result = value_counts_table(sales, "channel")

    counts = dict(zip(result["channel"], result["count"]))
    assert counts["Online"] == 2
    assert counts["Retail"] == 2
    assert counts["Partner"] == 1
    assert result.loc[result["channel"] == "Online", "pct"].iloc[0] == 40.0


def test_value_counts_table_respects_top_n(sales):
    assert len(value_counts_table(sales, "channel", top_n=2)) == 2


def test_value_counts_table_labels_missing_values(sales):
    result = value_counts_table(sales, "satisfaction")

    missing_row = result[result["satisfaction"] == "(missing)"]
    assert missing_row["count"].iloc[0] == 2


# ---------------------------------------------------------------------------
# missing_by_group
# ---------------------------------------------------------------------------
def test_missing_by_group_for_one_column(sales):
    result = missing_by_group(sales, "region", target_column="satisfaction")

    assert list(result["region"]) == ["West", "North", "South"]
    assert result.loc[result["region"] == "West", "missing_pct"].iloc[0] == 100.0
    assert result.loc[result["region"] == "South", "missing_pct"].iloc[0] == 0.0


def test_missing_by_group_across_all_columns(sales):
    result = missing_by_group(sales, "region")

    # 7 columns in the fixture, so West's single missing cell is 1/7 = 14.3%.
    assert result.loc[result["region"] == "West", "missing"].iloc[0] == 1
    assert result.loc[result["region"] == "West", "missing_pct"].iloc[0] == 14.3


# ---------------------------------------------------------------------------
# registry and dispatch
# ---------------------------------------------------------------------------
def test_list_tools_describes_every_tool():
    tools = list_tools()

    assert len(tools) == len(TOOL_REGISTRY) == 6
    for tool in tools:
        assert {"name", "description", "parameters"} <= set(tool)


def test_registry_functions_are_callable():
    for spec in TOOL_REGISTRY.values():
        assert callable(spec["function"])


def test_run_tool_dispatches_by_name(sales):
    result = run_tool("group_comparison", sales, group_column="region", value_column="revenue")

    assert list(result["region"]) == ["West", "South", "North"]


def test_run_tool_rejects_unknown_tool(sales):
    with pytest.raises(ToolError, match="Unknown tool"):
        run_tool("make_me_a_sandwich", sales)
