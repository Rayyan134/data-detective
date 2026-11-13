"""Tests for src.profiler.

Every test builds its own small dataset, so the suite is independent of any
file in data/ and runs in milliseconds. Run with:

    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.profiler import (
    DatasetError,
    column_summary,
    format_bytes,
    load_csv,
    numeric_summary,
    profile_dataset,
)


def write_csv(tmp_path, text: str, name: str = "data.csv"):
    """Write a CSV with UTF-8 encoding and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------
def test_load_csv_reads_path_and_strips_column_names(tmp_path):
    path = write_csv(tmp_path, " name , age \nAda,36\nGrace,45\n")

    df = load_csv(path)

    assert list(df.columns) == ["name", "age"]
    assert df.shape == (2, 2)


def test_load_csv_accepts_file_like_object(tmp_path):
    buffer = io.BytesIO(b"city,population\nDubai,3600000\n")

    df = load_csv(buffer)

    assert df.shape == (1, 2)
    assert df.loc[0, "city"] == "Dubai"


def test_load_csv_rewinds_partially_read_file():
    buffer = io.StringIO("a,b\n1,2\n")
    buffer.read(3)  # simulate a stream that has already been consumed

    df = load_csv(buffer)

    assert df.shape == (1, 2)


def test_load_csv_strips_utf8_bom():
    buffer = io.BytesIO("\ufeffname,score\nA,1\n".encode("utf-8"))

    df = load_csv(buffer)

    assert list(df.columns) == ["name", "score"]


def test_load_csv_rejects_none():
    with pytest.raises(DatasetError, match="No file"):
        load_csv(None)


def test_load_csv_rejects_empty_file(tmp_path):
    path = write_csv(tmp_path, "")

    with pytest.raises(DatasetError, match="empty"):
        load_csv(path)


def test_load_csv_rejects_file_without_data(tmp_path):
    path = write_csv(tmp_path, "\n\n\n")

    with pytest.raises(DatasetError, match="no data"):
        load_csv(path)


def test_load_csv_rejects_non_utf8(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes(b"name\nJos\xe9\n")

    with pytest.raises(DatasetError, match="UTF-8"):
        load_csv(path)


def test_load_csv_rejects_too_many_rows(tmp_path):
    path = write_csv(tmp_path, "a\n1\n2\n3\n")

    with pytest.raises(DatasetError, match="more than 2 rows"):
        load_csv(path, max_rows=2)


def test_load_csv_rejects_too_many_columns(tmp_path):
    path = write_csv(tmp_path, "a,b,c\n1,2,3\n")

    with pytest.raises(DatasetError, match="more than 2 columns"):
        load_csv(path, max_columns=2)


def test_load_csv_allows_header_with_no_rows(tmp_path):
    path = write_csv(tmp_path, "a,b\n")

    df = load_csv(path)

    assert df.shape == (0, 2)


# ---------------------------------------------------------------------------
# load_csv: a message a user can act on
# ---------------------------------------------------------------------------
def test_dataset_error_is_a_value_error():
    # Callers can catch the specific error or the general one.
    assert issubclass(DatasetError, ValueError)


# ---------------------------------------------------------------------------
# column_summary
# ---------------------------------------------------------------------------
def test_column_summary_counts_missing_values():
    df = pd.DataFrame({"score": [1.0, None, 3.0, None], "label": ["a", "b", "a", "c"]})

    summary = column_summary(df)
    score = summary.set_index("column").loc["score"]

    assert summary.shape[0] == 2
    assert score["missing"] == 2
    assert score["missing_pct"] == 50.0
    assert score["unique"] == 2
    assert score["non_null"] == 2


# ---------------------------------------------------------------------------
# numeric_summary
# ---------------------------------------------------------------------------
def test_numeric_summary_computes_statistics():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "name": ["a", "b", "c", "d"]})

    summary = numeric_summary(df)
    row = summary.loc["value"]

    assert list(summary.index) == ["value"]  # text column excluded
    assert row["count"] == 4
    assert row["mean"] == 2.5
    assert row["median"] == 2.5
    assert row["min"] == 1.0
    assert row["max"] == 4.0


def test_numeric_summary_handles_no_numeric_columns():
    df = pd.DataFrame({"name": ["a", "b"]})

    summary = numeric_summary(df)

    assert summary.empty
    assert list(summary.columns) == ["count", "mean", "std", "min", "q1", "median", "q3", "max"]


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------
def test_profile_dataset_reports_shape_and_missingness():
    df = pd.DataFrame(
        {
            "value": [1.0, None, 3.0, 4.0],
            "label": ["a", "b", None, "d"],
        }
    )

    profile = profile_dataset(df)

    assert profile["n_rows"] == 4
    assert profile["n_columns"] == 2
    assert profile["total_missing"] == 2
    assert profile["missing_pct"] == 25.0
    assert profile["duplicate_rows"] == 0


def test_profile_dataset_counts_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    profile = profile_dataset(df)

    assert profile["duplicate_rows"] == 1


def test_profile_dataset_warns_about_quality_issues():
    df = pd.DataFrame(
        {
            "mostly_missing": [None, None, "x", None],
            "constant": [1, 1, 1, 1],
            "Unnamed: 0": [0, 1, 2, 3],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    warnings = " ".join(profile_dataset(df)["warnings"])

    assert "mostly_missing" in warnings and "75.0% missing" in warnings
    assert "constant" in warnings and "only one unique value" in warnings
    assert "no header" in warnings


def test_profile_dataset_warns_when_there_are_no_rows():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64")})

    profile = profile_dataset(df)

    assert profile["n_rows"] == 0
    assert any("no rows" in warning for warning in profile["warnings"])


def test_profile_dataset_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        profile_dataset([1, 2, 3])


# ---------------------------------------------------------------------------
# format_bytes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [(512, "512.0 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
)
def test_format_bytes(num_bytes, expected):
    assert format_bytes(num_bytes) == expected
