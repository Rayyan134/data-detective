# Data Detective

**An Autonomous AI Data Investigation Agent**

Data Detective is an AI-powered application for investigating datasets with natural-language questions. Upload a CSV, ask a question, and receive findings backed by calculations performed in Python — not invented by a language model.

> **Status:** Day 2 of the DDS build. CSV loading, dataset profiling and the upload interface are working and tested. The LLM planning layer is not connected yet.

## What works today

- Upload a CSV file through the Streamlit interface.
- Input validation: empty files, non-UTF-8 files, malformed CSV, oversized datasets.
- Dataset profiling computed with Pandas: shape, data types, missing values, unique counts, most common values, duplicate rows, numeric summary statistics.
- Automatic data-quality warnings (columns with heavy missingness, constant columns, unnamed columns).
- Unit tests for the profiling module (`pytest`).

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
│   ├── investigator.py     # Planned: agent planning and tool selection
│   └── visualizer.py       # Planned: charts from computed results
├── tests/
│   └── test_profiler.py    # Unit tests for src/profiler.py
├── data/                   # Local sample datasets (never commit private data)
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
- [ ] Predefined analytical tools (comparison, correlation, trend, anomaly)
- [ ] LLM planning with structured tool selection
- [ ] Evidence validation and investigation report
- [ ] Visualisations
- [ ] Deployment

## Challenge

Developed as part of the **Decoding Data Science (DDS) Academy AI Application Building Challenge — November 2026**.
