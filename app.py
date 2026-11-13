"""Data Detective - Streamlit interface.

Run with:
    .\\.venv\\Scripts\\Activate.ps1
    streamlit run app.py

This file is the presentation layer only. All calculations happen in
``src/profiler.py`` so that the analytics stay testable and reusable.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.profiler import (
    MAX_COLUMNS,
    MAX_ROWS,
    DatasetError,
    column_summary,
    format_bytes,
    load_csv,
    numeric_summary,
    profile_dataset,
)

st.set_page_config(
    page_title="Data Detective",
    page_icon="🔍",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV from raw bytes.

    ``st.cache_data`` skips re-reading the file when Streamlit reruns the
    script (which it does on every interaction). Caching by bytes is safe:
    different files produce different hashes.
    """
    return load_csv(io.BytesIO(file_bytes))


def render_profile(profile: dict, df: pd.DataFrame) -> None:
    """Draw the summary metrics, warnings and detail tabs."""
    columns = st.columns(4)
    columns[0].metric("Rows", f"{profile['n_rows']:,}")
    columns[1].metric("Columns", f"{profile['n_columns']:,}")
    columns[2].metric(
        "Missing cells",
        f"{profile['total_missing']:,}",
        help=f"{profile['missing_pct']}% of all cells are empty.",
    )
    columns[3].metric(
        "Duplicate rows",
        f"{profile['duplicate_rows']:,}",
        help="Rows that are exact copies of another row.",
    )

    st.caption(
        f"In-memory size: {format_bytes(profile['memory_bytes'])}. "
        "All values below are computed by pandas from your uploaded file."
    )

    if profile["warnings"]:
        st.warning(
            "Data-quality warnings\n\n"
            + "\n".join(f"- {warning}" for warning in profile["warnings"])
        )
    else:
        st.success("No data-quality warnings were detected in this dataset.")

    preview_tab, columns_tab, numeric_tab, quality_tab = st.tabs(
        ["Preview", "Columns", "Numeric summary", "Data quality"]
    )

    with preview_tab:
        st.caption("Showing the first 100 rows.")
        st.dataframe(df.head(100), width="stretch")

    with columns_tab:
        st.caption(
            "One row per column. 'unique' ignores missing values; "
            "'dtype' is the type pandas inferred."
        )
        st.dataframe(column_summary(df), width="stretch", hide_index=True)

    with numeric_tab:
        summary = numeric_summary(df)
        if summary.empty:
            st.info("This dataset has no numeric columns to summarise.")
        else:
            st.caption(
                "Descriptive statistics for numeric columns. "
                "q1, median and q3 are the 25th, 50th and 75th percentiles."
            )
            st.dataframe(summary, width="stretch")

    with quality_tab:
        st.markdown(
            f"""
- **Rows:** {profile['n_rows']:,}
- **Columns:** {profile['n_columns']:,}
- **Total cells:** {profile['n_rows'] * profile['n_columns']:,}
- **Missing cells:** {profile['total_missing']:,} ({profile['missing_pct']}%)
- **Duplicate rows:** {profile['duplicate_rows']:,}
"""
        )
        if profile["warnings"]:
            st.markdown("**Warnings**")
            for warning in profile["warnings"]:
                st.markdown(f"- {warning}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Data Detective profiles a CSV dataset before any investigation "
        "begins. The numbers come from pandas, not from a language model."
    )
    st.subheader("Input limits")
    st.markdown(
        f"- Max rows: **{MAX_ROWS:,}**\n"
        f"- Max columns: **{MAX_COLUMNS:,}**\n"
        "- Encoding: **UTF-8**\n"
        "- Format: **.csv**"
    )
    st.subheader("Privacy")
    st.write(
        "Files are processed in memory for this session only and are not "
        "written to disk."
    )
    st.divider()
    st.caption("Day 2 build: profiling is live. The AI agent is not connected yet.")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🔍 Data Detective")
st.write(
    "Upload a CSV dataset to inspect its structure, missing values, data types "
    "and distributions."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=f"UTF-8 CSV, up to {MAX_ROWS:,} rows and {MAX_COLUMNS:,} columns.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin. Nothing is sent anywhere.")
    with st.expander("What gets checked?"):
        st.markdown(
            """
- Can the file be read as UTF-8 CSV?
- How many rows and columns does it contain?
- Which columns have missing values, and how many?
- What data type did pandas infer for each column?
- Are there duplicate rows, constant columns or unnamed columns?
"""
        )
    st.stop()

st.subheader(uploaded_file.name)
st.caption(f"Uploaded size: {format_bytes(uploaded_file.size)}")

try:
    data = load_uploaded_csv(uploaded_file.getvalue())
except DatasetError as error:
    st.error(f"**Could not read this file.** {error}")
    st.stop()

render_profile(profile_dataset(data), data)
