"""Data Detective - Streamlit interface.

Run with:
    .\\.venv\\Scripts\\Activate.ps1
    streamlit run app.py

This file is the presentation layer only. All calculations happen in
``src/profiler.py`` and ``src/investigator.py``, so the analytics stay
testable and reusable.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audit import log_investigation
from src.investigator import investigate
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
from src.ui_report import (
    dataset_label,
    example_questions,
    finding_sections,
    interpretation_section,
    plan_lines,
    redaction_candidates,
    status_banner,
    unverified_warning,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_sales.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "investigations.jsonl"

st.set_page_config(
    page_title="Data Detective",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "data": None,
        "dataset_name": "",
        "dataset_key": None,
        "profile": None,
        "report": None,
        "question": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# ---------------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------------
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


def render_investigation(report) -> None:
    """Render an InvestigationReport: status, plan, findings, answer, caution."""
    banner = status_banner(report)
    level = banner["level"]
    st.markdown(f"### {banner['title']}")
    getattr(st, level)(banner["message"])

    plan = plan_lines(report)
    if plan:
        with st.expander("Plan (AI-generated, then validated)", expanded=True):
            for index, line in enumerate(plan, start=1):
                st.markdown(f"{index}. {line}")
            if report.plan is not None and report.plan.repairs:
                st.markdown(
                    "**Normalised parameters** (trailing text trimmed to a known "
                    "value):"
                )
                for note in report.plan.repairs:
                    st.markdown(f"- {note}")

    sections = finding_sections(report)
    if sections:
        st.markdown("#### Findings (computed in Python)")
        for section in sections:
            st.markdown(f"**{section['title']}**")
            if section["purpose"]:
                st.caption(section["purpose"])
            st.caption(
                "Parameters: `"
                + json.dumps(section["params"], ensure_ascii=False, sort_keys=True)
                + "`"
            )
            if section["frame"] is not None:
                st.dataframe(section["frame"], width="stretch")
            else:
                st.error(f"Tool failed: {section['error']}")
            if section["table"]:
                with st.expander("Markdown table"):
                    st.code(section["table"], language="markdown")

    narrative = interpretation_section(report)
    if narrative is not None:
        st.markdown("#### Answer (AI interpretation — not independently verified)")
        if narrative["answer"]:
            st.markdown(narrative["answer"])
        if narrative["interpretation"]:
            st.markdown(narrative["interpretation"])
        if narrative["confidence"]:
            st.caption(f"Confidence: {narrative['confidence']}")
        for caveat in narrative["caveats"]:
            st.markdown(f"- {caveat}")

    caution = unverified_warning(report)
    if caution:
        st.warning(caution)

    with st.expander("Full report (Markdown)"):
        st.code(report.to_markdown(), language="markdown")
    st.download_button(
        "Download report (.md)",
        report.to_markdown(),
        file_name="investigation.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Data Detective profiles a CSV dataset and investigates it with an "
        "AI agent. Every number in the findings is computed by Python; the "
        "model only plans the analysis and writes the narrative."
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
        "written to disk. Activity is logged as metadata only (no data values, "
        "no API key)."
    )
    st.divider()
    st.caption("Day 5 build: the investigator is connected to the interface.")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🔍 Data Detective")
st.write(
    "Upload a CSV dataset to inspect it, then ask a question in plain English. "
    "The agent plans the analysis, runs predefined tools and reports "
    "evidence-backed findings."
)

_init_state()

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=f"UTF-8 CSV, up to {MAX_ROWS:,} rows and {MAX_COLUMNS:,} columns.",
)

sample_col, upload_hint = st.columns([1, 3])
with sample_col:
    use_sample = st.button("Use the sample dataset", key="use_sample")
with upload_hint:
    st.caption("No file handy? Load the bundled sample_sales.csv to try everything.")

# --- dataset loading -------------------------------------------------------
if use_sample:
    try:
        st.session_state["data"] = load_csv(SAMPLE_DATASET)
        st.session_state["dataset_name"] = SAMPLE_DATASET.name
        st.session_state["dataset_key"] = ("sample", SAMPLE_DATASET.stat().st_size)
        st.session_state["profile"] = None
        st.session_state["report"] = None
        st.session_state["question"] = ""
    except DatasetError as error:
        st.error(f"**Could not load the sample dataset.** {error}")

if uploaded_file is not None:
    file_key = (uploaded_file.name, uploaded_file.size)
    if file_key != st.session_state["dataset_key"]:
        try:
            data = load_uploaded_csv(uploaded_file.getvalue())
            st.session_state["data"] = data
            st.session_state["dataset_name"] = uploaded_file.name
            st.session_state["dataset_key"] = file_key
            st.session_state["profile"] = None
            st.session_state["report"] = None
            st.session_state["question"] = ""
        except DatasetError as error:
            st.error(f"**Could not read this file.** {error}")

data = st.session_state["data"]

if data is None:
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

# --- dataset loaded -------------------------------------------------------
st.subheader(st.session_state["dataset_name"])
st.caption(dataset_label(st.session_state["dataset_name"], data))

if st.session_state["profile"] is None:
    st.session_state["profile"] = profile_dataset(data)
profile = st.session_state["profile"]

render_profile(profile, data)

# --- investigation ---------------------------------------------------------
st.divider()
st.subheader("Ask a question about this dataset")
st.caption(
    "The model plans; Python computes. Findings and the AI answer are shown "
    "separately below."
)

suggestions = example_questions(data)
if suggestions:
    st.caption("Try one of these:")
    suggestion_cols = st.columns(len(suggestions))
    for column, question in zip(suggestion_cols, suggestions):
        if column.button(question, key=f"example_{question}", use_container_width=True):
            st.session_state["question"] = question
            st.rerun()

question = st.text_input(
    "Your question",
    key="question",
    placeholder="e.g. Which region has the highest average revenue?",
    help="Ask in plain English. The agent handles the rest.",
)

redact_options = [str(c) for c in data.columns]
suggested_redactions = redaction_candidates(data)
redact = st.multiselect(
    "Columns to keep away from the model (optional)",
    options=redact_options,
    default=suggested_redactions,
    help=(
        "These columns are still described by name and type, but no value from "
        "them reaches the model or the evidence. Pre-selected columns look like "
        "identifiers."
    ),
)

run_disabled = not str(question).strip()
run_clicked = st.button(
    "Investigate", key="investigate", type="primary", disabled=run_disabled
)

if run_clicked and not run_disabled:
    started = time.perf_counter()
    with st.spinner("Planning the investigation, running the tools and writing the answer…"):
        report = investigate(
            str(question).strip(),
            data,
            profile=profile,
            redact_columns=redact or None,
        )
    latency_ms = int((time.perf_counter() - started) * 1000)
    st.session_state["report"] = report
    log_investigation(
        LOG_PATH,
        question=str(question).strip(),
        report=report,
        dataset_label=st.session_state["dataset_name"],
        n_rows=len(data),
        n_columns=data.shape[1],
        redact_columns=redact,
        latency_ms=latency_ms,
    )

report = st.session_state["report"]
if report is not None:
    render_investigation(report)
    if st.button("Clear the result", key="clear_result"):
        st.session_state["report"] = None
        st.session_state["question"] = ""
        st.rerun()
