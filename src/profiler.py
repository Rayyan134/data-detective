"""Dataset loading and profiling.

This module is the analytical core of Data Detective. It contains no UI code
and no LLM code, which means it can be unit-tested directly and reused later
as a tool that the agent calls.

Public functions:
    load_csv(source)        -> validated pandas DataFrame
    profile_dataset(df)     -> dict of summary statistics and warnings
    column_summary(df)      -> one row per column
    numeric_summary(df)     -> descriptive statistics for numeric columns
    format_bytes(n)         -> human-readable size string
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd

# ---------------------------------------------------------------------------
# Limits. These keep profiling and rendering responsive for a demo.
# ---------------------------------------------------------------------------
MAX_ROWS = 100_000
MAX_COLUMNS = 200
HIGH_MISSING_THRESHOLD = 0.5  # 50% of a column's values missing

__all__ = [
    "DatasetError",
    "MAX_ROWS",
    "MAX_COLUMNS",
    "load_csv",
    "profile_dataset",
    "column_summary",
    "numeric_summary",
    "format_bytes",
]

# A CSV can come from a path (str / Path) or a file-like object such as
# Streamlit's UploadedFile or io.BytesIO.
CsvSource = str | Path | IO[Any]


class DatasetError(ValueError):
    """Raised when a file cannot be used as a dataset.

    Subclassing ValueError means callers can catch either this specific
    exception or a general ValueError.
    """


def load_csv(
    source: CsvSource | None,
    *,
    max_rows: int = MAX_ROWS,
    max_columns: int = MAX_COLUMNS,
) -> pd.DataFrame:
    """Read and validate a CSV file.

    Args:
        source: A file path or a file-like object containing CSV text.
        max_rows: Reject datasets with more rows than this.
        max_columns: Reject datasets with more columns than this.

    Returns:
        A DataFrame with cleaned column names.

    Raises:
        DatasetError: If the source is missing, empty, unreadable,
            unparsable, or larger than the configured limits.
    """
    if max_rows < 1 or max_columns < 1:
        raise ValueError("max_rows and max_columns must be positive integers.")

    if source is None:
        raise DatasetError("No file was provided.")

    if _source_size(source) == 0:
        raise DatasetError("The uploaded file is empty.")

    # Streamlit may hand us a file object that has already been read from.
    # Rewinding avoids mysterious "no data" errors.
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except (OSError, ValueError) as exc:  # pragma: no cover - rare
            raise DatasetError("The file could not be read.") from exc

    try:
        # nrows + 1 lets us detect "too many rows" without reading the whole
        # file. utf-8-sig transparently strips a byte-order mark if present.
        df = pd.read_csv(source, nrows=max_rows + 1, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(
            "The file contains no data. It may be empty or contain only blank lines."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(
            "The file could not be parsed as CSV. Check the delimiter and quoting."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(
            "The file is not valid UTF-8 text. Re-save it as a UTF-8 CSV."
        ) from exc
    except (OSError, ValueError) as exc:
        raise DatasetError(f"The file could not be read: {exc}") from exc

    if len(df) > max_rows:
        raise DatasetError(
            f"The dataset has more than {max_rows:,} rows. "
            "Please upload a smaller sample."
        )
    if df.shape[1] == 0:
        raise DatasetError("The file contains no columns.")
    if df.shape[1] > max_columns:
        raise DatasetError(
            f"The dataset has more than {max_columns:,} columns. "
            "Please upload a smaller selection."
        )

    # Column names sometimes arrive with stray spaces (" age"), which makes
    # later lookups confusing. Strip them once, at the boundary.
    df.columns = [str(column).strip() for column in df.columns]
    return df.reset_index(drop=True)


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per column.

    The result is a plain DataFrame so it can be displayed directly in
    Streamlit or asserted on in tests.
    """
    _require_dataframe(df)

    rows: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": str(name),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": missing,
                "missing_pct": round(missing / len(series) * 100, 1) if len(series) else 0.0,
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns.

    Uses ``DataFrame.describe`` and transposes it so each row is a column.
    Returns an empty DataFrame (with the expected columns) if the dataset has
    no numeric columns.
    """
    _require_dataframe(df)

    expected_columns = ["count", "mean", "std", "min", "q1", "median", "q3", "max"]
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=expected_columns).rename_axis("column")

    summary = numeric.describe().T
    summary = summary.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    summary = summary[expected_columns]
    summary.index.name = "column"
    return summary.round(3)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a profile of the dataset.

    Returns a dictionary of summary values plus a list of plain-language
    data-quality warnings. Every value is computed from the DataFrame, so the
    numbers are reproducible and can be shown as evidence.
    """
    _require_dataframe(df)

    n_rows, n_columns = df.shape
    per_column = column_summary(df)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = n_rows * n_columns
    total_missing = int(per_column["missing"].sum()) if n_columns else 0

    warnings: list[str] = []
    if n_rows == 0:
        warnings.append("The dataset has columns but no rows.")
    if n_columns == 0:
        warnings.append("The dataset has no columns.")
    for record in per_column.to_dict("records"):
        name = record["column"]
        if record["missing_pct"] >= HIGH_MISSING_THRESHOLD * 100:
            warnings.append(
                f"Column '{name}' is {record['missing_pct']}% missing."
            )
        if n_rows and record["unique"] <= 1 and record["non_null"] > 0:
            warnings.append(
                f"Column '{name}' is constant (only one unique value)."
            )
        if name.startswith("Unnamed:"):
            warnings.append(
                f"Column '{name}' has no header. Rename it before analysis."
            )
    if duplicate_rows:
        warnings.append(f"{duplicate_rows:,} duplicate rows detected.")

    memory_bytes = int(df.memory_usage(deep=True).sum())
    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "memory_bytes": memory_bytes,
        "memory_mb": round(memory_bytes / (1024**2), 2),
        "duplicate_rows": duplicate_rows,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / total_cells * 100, 1) if total_cells else 0.0,
        "warnings": warnings,
    }


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as B, KB, MB or GB."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - loop always returns


def _require_dataframe(df: pd.DataFrame) -> None:
    """Fail early with a clear message if the input is not a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")


def _source_size(source: CsvSource | None) -> int | None:
    """Best-effort size in bytes for a path or file-like object.

    Returns None when the size cannot be determined, in which case we simply
    let pandas decide whether the file is readable.
    """
    if isinstance(source, (str, Path)):
        try:
            return Path(source).stat().st_size
        except OSError:
            return None

    size = getattr(source, "size", None)
    if isinstance(size, int):
        return size

    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        try:
            return len(getvalue())
        except (OSError, ValueError):
            return None
    return None
