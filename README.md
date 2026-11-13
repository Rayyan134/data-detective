# Data Detective

**An Autonomous AI Data Investigation Agent**

Data Detective is an AI-powered application for investigating datasets with natural-language questions. Upload a CSV, ask a question, and receive findings backed by calculations performed in Python — not invented by a language model.

> **Status:** Day 3 of the DDS build. CSV loading, profiling, six analytical tools and a 15-question evaluation set are working and tested (47 tests). The LLM planning layer is not connected yet.

## What works today

- Upload a CSV file through the Streamlit interface.
- Input validation: empty files, non-UTF-8 files, malformed CSV, oversized datasets.
- Dataset profiling computed with Pandas: shape, data types, missing values, unique counts, most common values, duplicate rows, numeric summary statistics.
- Automatic data-quality warnings (columns with heavy missingness, constant columns, unnamed columns).
- Six deterministic analytical tools (see below) exposed through a tool registry.
- A 15-question evaluation set with hand-checked expected answers (`eval/questions.json`).
- Unit tests for the profiling and tools modules (`pytest`, 47 tests).

## Analytical tools

These are the agent's "hands": plain functions that compute results from the data. Each one validates its inputs and raises `ToolError` with a readable message when the request cannot be answered.

| Tool | Question it answers |
|---|---|
| `group_comparison` | How does a numeric value compare across categories? (mean, median or sum) |
| `correlation` | Do two numeric columns move together? |
| `trend_over_time` | How does a numeric value change by day, week or month? |
| `find_outliers` | Which values are unusually high or low? (IQR method) |
| `value_counts_table` | How often does each value appear? |
| `missing_by_group` | Where is data missing, and in which groups? |

`TOOL_REGISTRY` describes each tool and its parameters in a machine-readable form, and `run_tool(name, df, **params)` executes one by name. This is the interface the investigator will hand to the language model, and it is also why the tools are separate from the UI and the LLM.

## Planned features

- **Autonomous investigation:** interpret a question, plan steps and select predefined analytical tools.
- **Visualisations:** charts built from actual computed results.
- **Evidence-backed insights:** explanations that cite the calculations behind them.
- **Structured report:** plan, findings, evidence and limitations in one view.

## Project goals

- Make exploratory data analysis more accessible.
- Explore practical applications of agentic AI.
- Build an agent that plans and executes multi-step analytical tasks.
- Prioritise reliability, transparency and explainability over impressive-looking output.

## Technology stack

| Technology | Purpose | Status |
|---|---|---|
| Python 3.12 | Core application | In use |
| Streamlit | Web interface | In use |
| Pandas | Data manipulation and profiling | In use |
| NumPy | Numerical operations | In use |
| pytest | Unit tests | In use (development only) |
| python-dotenv | Loading local secrets from `.env` | Installed, not used yet |
| Matplotlib | Charts | Planned |
| LLM API | Question interpretation and planning | Planned (provider undecided) |

## Project structure

```
data-detective/
├── app.py                  # Streamlit interface (upload + profile view)
├── src/
│   ├── __init__.py         # Marks src/ as a Python package
│   ├── profiler.py         # CSV validation + profiling (no UI, no LLM)
│   ├── tools.py            # Predefined analytical tools + registry
│   ├── investigator.py     # Planned: agent planning and tool selection
│   └── visualizer.py       # Planned: charts from computed results
├── tests/
│   ├── test_profiler.py    # Unit tests for src/profiler.py
│   └── test_tools.py       # Unit tests for src/tools.py
├── eval/
│   └── questions.json      # Fixed question set with expected answers
├── data/                   # Sample datasets (never commit private data)
│   ├── sample_people.csv   # Small synthetic set with missing/duplicate values
│   └── sample_sales.csv    # 180 synthetic orders with outliers and gaps
├── requirements.txt        # Pinned runtime dependencies
├── requirements-dev.txt    # Test-only dependencies
├── .env.example            # Template for local secrets (copy to .env)
└── README.md
```

The separation matters: `src/profiler.py` contains no Streamlit code and no LLM code, so it can be tested directly and reused later as an agent tool.

## Requirements

- Python 3.12
- Git

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/Rayyan134/data-detective.git
cd data-detective
uv venv --python 3.12
uv pip install -r requirements.txt
```

Without `uv`, use the standard library instead:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app opens at <http://localhost:8501>. Stop it with `Ctrl+C`.

## Run the tests

```powershell
uv pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Configuration

No configuration is required for profiling. When the LLM layer is added, API keys will be read from a local `.env` file:

```powershell
Copy-Item .env.example .env
```

`.env` is git-ignored. Only `.env.example` (with placeholder values) is committed.

## Input limits

| Limit | Value | Reason |
|---|---|---|
| Maximum rows | 100,000 | Keeps analysis and rendering responsive |
| Maximum columns | 200 | Keeps profiling and the UI readable |
| Encoding | UTF-8 (with or without BOM) | Predictable text handling |
| Format | `.csv` | The supported input format |

Files are processed in memory. Uploaded datasets are not written to disk or permanently stored.

## Design principles

1. **Python computes, the LLM explains.** Every number shown to a user must come from a Python calculation.
2. **No fabricated results.** If a calculation cannot be performed, the app says so.
3. **Fail clearly.** Invalid input produces a specific, readable error.
4. **Small, testable modules.** Data logic lives in `src/`, away from the UI.
5. **No unnecessary frameworks.** A single agent with reliable tools, not a multi-agent system.

## Data safety

- Do not upload confidential or personally identifiable information during development.
- Validate uploaded files before processing.
- Do not permanently store user datasets.
- Use public datasets in accordance with their individual licences.
- Never commit datasets or API credentials.

## Roadmap

- [x] Environment, repository and dependency pinning
- [x] CSV loading and dataset profiling with tests
- [x] Streamlit upload and profile interface
- [x] Predefined analytical tools (comparison, correlation, trend, outliers) with tests
- [x] Fixed evaluation question set with expected answers
- [ ] LLM planning with structured tool selection
- [ ] Evidence validation and investigation report
- [ ] Visualisations
- [ ] Deployment

## Challenge

Developed as part of the **Decoding Data Science (DDS) Academy AI Application Building Challenge — November 2026**.
