# Data Detective

**An Autonomous AI Data Investigation Agent**

Data Detective is an AI-powered application for investigating datasets with natural-language questions. Upload a CSV, ask a question, and receive findings backed by calculations performed in Python — not invented by a language model.

> **Status:** Day 3 of the DDS build. CSV loading, profiling, six analytical tools, a 15-question evaluation set and an isolated, live-tested Gemini REST client are working (100 tests). The agent planning layer is next.

## What works today

- Upload a CSV file through the Streamlit interface.
- Input validation: empty files, non-UTF-8 files, malformed CSV, oversized datasets.
- Dataset profiling computed with Pandas: shape, data types, missing values, unique counts, most common values, duplicate rows, numeric summary statistics.
- Automatic data-quality warnings (columns with heavy missingness, constant columns, unnamed columns).
- Six deterministic analytical tools (see below) exposed through a tool registry.
- A 15-question evaluation set with hand-checked expected answers (`eval/questions.json`).
- An isolated Gemini Flash client (`src/llm_client.py`) using plain HTTP, with typed errors for authentication, rate limits, timeouts and malformed responses, plus retries on transient overload.
- Unit tests for the profiling, tools and LLM client modules (`pytest`, 100 tests).

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
| python-dotenv | Loading local secrets from `.env` | In use |
| Matplotlib | Charts | Planned |
| Gemini Flash (REST) | Question interpretation and planning | Client built; planning layer next |

## Project structure

```
data-detective/
├── app.py                  # Streamlit interface (upload + profile view)
├── src/
│   ├── __init__.py         # Marks src/ as a Python package
│   ├── profiler.py         # CSV validation + profiling (no UI, no LLM)
│   ├── tools.py            # Predefined analytical tools + registry
│   ├── llm_client.py       # Isolated Gemini REST client (the only LLM code)
│   ├── investigator.py     # Planned: agent planning and tool selection
│   └── visualizer.py       # Planned: charts from computed results
├── tests/
│   ├── test_profiler.py    # Unit tests for src/profiler.py
│   ├── test_tools.py       # Unit tests for src/tools.py
│   ├── test_eval_set.py    # Guards eval/questions.json against drift
│   └── test_llm_client.py  # Unit tests for src/llm_client.py (no network)
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

Profiling and the analytical tools need no configuration. The language model layer reads its settings from a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and replace `replace-me` with a real Gemini API key. Get a free key at <https://aistudio.google.com/api-keys>.

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes, for LLM features | Authenticates requests to the Gemini API |
| `GEMINI_MODEL` | No | Overrides the model. Defaults to `gemini-3.6-flash` |
| `GEMINI_TIMEOUT` | No | Seconds to wait per request. Defaults to `60` |

`.env` is git-ignored. Only `.env.example` (with placeholder values) is committed. The key is read from the environment and is never written into the code or an error message.

### Test the Gemini connection

Before any agent behaviour is built, confirm the key, model and network work with one small round trip:

```powershell
.\.venv\Scripts\python.exe -m src.llm_client
```

The script prints the endpoint, model and timeout, lists the models the key can access, sends a one-line prompt and reports the reply, latency and token usage. It exits non-zero with a specific message if the key is missing, rejected, rate-limited or if the response is malformed.

> **Why the timeout and the retries.** `gemini-3.6-flash` is a *reasoning* model: it thinks before answering, and the default thinking level is `MEDIUM`. On the free tier even a one-line reply can take a few seconds. The connection test therefore requests `thinkingLevel: "LOW"`, which cuts latency, and allows 60 seconds. Transient `429` and `5xx` responses are retried twice with a short backoff before failing.
>
> **Why `gemini-3.6-flash` and not `gemini-3.8-flash`.** The newest Flash returned HTTP 503 (overloaded) on the free tier during setup, while 3.6 Flash served reliably. Both are free of charge. Change `GEMINI_MODEL` if you want to try the newer model later.

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
- [x] Isolated Gemini REST client with typed error handling and tests
- [ ] LLM planning with structured tool selection
- [ ] Evidence validation and investigation report
- [ ] Visualisations
- [ ] Deployment

## Challenge

Developed as part of the **Decoding Data Science (DDS) Academy AI Application Building Challenge — November 2026**.
