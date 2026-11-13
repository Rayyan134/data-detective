"""Predefined analytical tools.

These are the agent's "hands". Each function takes a DataFrame plus explicit
parameters and returns a computed pandas result. Nothing here calls an LLM,
and nothing here invents numbers: every value is calculated from the data.

The registry at the bottom describes each tool in a machine-readable form so
the investigator can advertise the available tools to a language model and
then execute the chosen one by name.

Public functions:
    group_comparison(df, group_column, value_column, metric) -> DataFrame
    correlation(df, columns, method)                         -> DataFrame
    trend_over_time(df, date_column, value_column, freq)     -> DataFrame
    find_outliers(df, column, multiplier)                    -> DataFrame
    value_counts_table(df, column, top_n)                    -> DataFrame
    missing_by_group(df, group_column, target_column)        -> DataFrame
    list_tools()                                             -> list[dict]
    run_tool(name, df, **params)                             -> DataFrame
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

__all__ = [
    "ToolError",
    "group_comparison",
    "correlation",
    "trend_over_time",
    "find_outliers",
    "value_counts_table",
    "missing_by_group",
    "TOOL_REGISTRY",
    "list_tools",
    "run_tool",
]


class ToolError(ValueError):
    """Raised when a tool cannot run because its input is unsuitable.

    Like ``DatasetError``, this subclasses ValueError so callers can catch the
    specific type or the general one.
    """


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------
def _require_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")


def _require_columns(df: pd.DataFrame, *columns: str) -> None:
    """Ensure every requested column exists, with a helpful message if not."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(str(column) for column in df.columns)
        raise ToolError(
            f"Column(s) not found: {', '.join(missing)}. "
            f"Available columns: {available}."
        )


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ToolError(
            f"Column '{column}' is not numeric (dtype: {df[column].dtype}). "
            "Choose a numeric column."
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def group_comparison(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    metric: str = "mean",
) -> pd.DataFrame:
    """Compare a numeric value across the categories of another column.

    Example: average revenue per region.

    Args:
        df: The dataset.
        group_column: Column whose categories define the groups.
        value_column: Numeric column to summarise within each group.
        metric: Which statistic to sort by: "mean", "median" or "sum".

    Returns:
        One row per group with count, mean, median and sum, sorted by `metric`
        from highest to lowest.
    """
    _require_dataframe(df)
    _require_columns(df, group_column, value_column)
    _require_numeric(df, value_column)
    if metric not in {"mean", "median", "sum"}:
        raise ToolError("metric must be one of: mean, median, sum.")

    grouped = (
        df.groupby(group_column, dropna=False)[value_column]
        .agg(["count", "mean", "median", "sum"])
        .round(3)
        .reset_index()
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a correlation matrix for numeric columns.

    Correlation runs from -1 (opposite) through 0 (no linear relationship) to
    +1 (together). It describes association, never causation.

    Args:
        df: The dataset.
        columns: Optional subset of numeric columns. All numeric columns when
            omitted.
        method: "pearson", "spearman" or "kendall".

    Returns:
        A square DataFrame of correlation coefficients.
    """
    _require_dataframe(df)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ToolError("method must be one of: pearson, spearman, kendall.")

    if columns is None:
        numeric = df.select_dtypes(include="number")
    else:
        _require_columns(df, *columns)
        for column in columns:
            _require_numeric(df, column)
        numeric = df[list(columns)]

    if numeric.shape[1] < 2:
        raise ToolError("Correlation needs at least two numeric columns.")

    return numeric.corr(method=method).round(3)


def trend_over_time(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """Aggregate a numeric column into time periods and measure the change.

    Args:
        df: The dataset.
        date_column: Column holding dates (parsed with pandas).
        value_column: Numeric column to total per period.
        freq: Pandas frequency alias, e.g. "D" (day), "W" (week), "ME"
            (month end).

    Returns:
        One row per period with total, order count, absolute change and
        percentage change.
    """
    _require_dataframe(df)
    _require_columns(df, date_column, value_column)
    _require_numeric(df, value_column)

    # format="mixed" tells pandas to infer the format per value, which avoids
    # the slow dateutil fallback and its noisy warning on odd values.
    parsed = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    invalid = int(parsed.isna().sum())
    if invalid:
        raise ToolError(
            f"{invalid} value(s) in '{date_column}' could not be parsed as dates."
        )

    work = pd.DataFrame({"period": parsed, "value": df[value_column]})
    try:
        grouped = work.set_index("period").resample(freq)["value"].agg(["sum", "count"])
    except ValueError as exc:
        raise ToolError(f"'{freq}' is not a valid pandas frequency alias.") from exc

    grouped = grouped.rename(columns={"sum": "total", "count": "orders"})
    grouped["change"] = grouped["total"].diff()
    grouped["pct_change"] = (grouped["total"].pct_change() * 100).round(1)

    result = grouped.round(3).reset_index()
    result["period"] = result["period"].dt.strftime("%Y-%m-%d")
    return result


def find_outliers(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Find values that fall far outside the usual range (IQR method).

    The interquartile range (IQR) is the middle 50% of values. Anything below
    ``q1 - multiplier * IQR`` or above ``q3 + multiplier * IQR`` is flagged.

    Args:
        df: The dataset.
        column: Numeric column to inspect.
        multiplier: Sensitivity. 1.5 is the standard; higher flags less.

    Returns:
        The outlier rows plus the lower and upper bounds used, sorted by the
        inspected column from highest to lowest.
    """
    _require_dataframe(df)
    _require_columns(df, column)
    _require_numeric(df, column)
    if multiplier <= 0:
        raise ToolError("multiplier must be greater than zero.")

    values = df[column].dropna()
    if values.empty:
        raise ToolError(f"Column '{column}' has no values to analyse.")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
    result = df.loc[mask].copy()
    result["iqr_lower_bound"] = round(float(lower), 3)
    result["iqr_upper_bound"] = round(float(upper), 3)
    return result.sort_values(column, ascending=False).reset_index(drop=True)


def value_counts_table(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Count how often each value appears in a column.

    Args:
        df: The dataset.
        column: Column to count (text or numeric).
        top_n: Maximum number of rows to return.

    Returns:
        A table of value, count and percentage of all rows, most common first.
        Missing values appear as the literal label "(missing)".
    """
    _require_dataframe(df)
    _require_columns(df, column)
    if top_n < 1:
        raise ToolError("top_n must be at least 1.")

    counts = df[column].value_counts(dropna=False)
    total = len(df)
    labels = ["(missing)" if pd.isna(value) else value for value in counts.index]

    result = pd.DataFrame(
        {
            column: labels,
            "count": counts.to_numpy(),
        }
    )
    result["pct"] = (result["count"] / total * 100).round(1)
    return result.head(top_n).reset_index(drop=True)


def missing_by_group(
    df: pd.DataFrame,
    group_column: str,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Measure missingness within each group.

    Args:
        df: The dataset.
        group_column: Column whose categories define the groups.
        target_column: Optional single column to check. When omitted, all
            columns are checked and the percentage is over all cells.

    Returns:
        One row per group with row count, missing count and missing percentage,
        sorted from most to least missing.
    """
    _require_dataframe(df)
    _require_columns(df, group_column)
    if target_column is not None:
        _require_columns(df, target_column)
        missing_per_row = df[target_column].isna().astype(int)
        denominator = 1
    else:
        missing_per_row = df.isna().sum(axis=1)
        denominator = df.shape[1]

    work = pd.DataFrame(
        {
            group_column: df[group_column],
            "_missing": missing_per_row,
        }
    )
    grouped = (
        work.groupby(group_column, dropna=False)["_missing"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"sum": "missing", "count": "rows"})
    grouped["missing_pct"] = (
        grouped["missing"] / (grouped["rows"] * denominator) * 100
    ).round(1)
    return grouped.sort_values("missing_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Registry: what the agent is allowed to call
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "group_comparison": {
        "function": group_comparison,
        "description": (
            "Compare a numeric column across the categories of another column "
            "(for example average revenue per region)."
        ),
        "parameters": {
            "group_column": {"type": "string", "required": True},
            "value_column": {"type": "string", "required": True},
            "metric": {
                "type": "string",
                "required": False,
                "enum": ["mean", "median", "sum"],
                "default": "mean",
            },
        },
    },
    "correlation": {
        "function": correlation,
        "description": (
            "Measure the linear relationship between two or more numeric columns."
        ),
        "parameters": {
            "columns": {"type": "array", "items": "string", "required": False},
            "method": {
                "type": "string",
                "required": False,
                "enum": ["pearson", "spearman", "kendall"],
                "default": "pearson",
            },
        },
    },
    "trend_over_time": {
        "function": trend_over_time,
        "description": (
            "Total a numeric column by day, week or month and measure the change "
            "between periods."
        ),
        "parameters": {
            "date_column": {"type": "string", "required": True},
            "value_column": {"type": "string", "required": True},
            "freq": {
                "type": "string",
                "required": False,
                "default": "ME",
                "description": "Pandas alias: D, W, ME, QE, YE.",
            },
        },
    },
    "find_outliers": {
        "function": find_outliers,
        "description": (
            "Find unusually high or low values in a numeric column using the "
            "interquartile range method."
        ),
        "parameters": {
            "column": {"type": "string", "required": True},
            "multiplier": {
                "type": "number",
                "required": False,
                "default": 1.5,
            },
        },
    },
    "value_counts_table": {
        "function": value_counts_table,
        "description": "Count the most common values in a column.",
        "parameters": {
            "column": {"type": "string", "required": True},
            "top_n": {"type": "integer", "required": False, "default": 10},
        },
    },
    "missing_by_group": {
        "function": missing_by_group,
        "description": (
            "Measure how much data is missing within each group, either for one "
            "column or across the whole row."
        ),
        "parameters": {
            "group_column": {"type": "string", "required": True},
            "target_column": {"type": "string", "required": False},
        },
    },
}


def list_tools() -> list[dict[str, Any]]:
    """Return the tool catalogue for prompt building and documentation."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
        for name, spec in TOOL_REGISTRY.items()
    ]


def run_tool(name: str, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
    """Execute a registered tool by name.

    Raises:
        ToolError: If the tool is unknown or its parameters are invalid.
    """
    if name not in TOOL_REGISTRY:
        raise ToolError(
            f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}."
        )
    function: Callable[..., pd.DataFrame] = TOOL_REGISTRY[name]["function"]
    return function(df, **params)
