# Data Detective

**An Autonomous AI Data Investigation Agent**

Data Detective is an AI-powered application for investigating datasets with natural-language questions. Upload a CSV, ask a question, and receive findings backed by calculations performed in Python — not invented by a language model.

> **Status:** Day 7 of the DDS build, **deployed**. Everything from Day 6 works. Day 7 delivered the last planned feature — **visualisations** (charts built from computed results, via `src/visualizer.py`) — plus deployment to Streamlit Community Cloud. Live app: https://data-detective-r.streamlit.app/. 281 tests pass offline. Free-tier quota (20 req/day/model) still blocks the authoritative primary-model evaluation run (backup-model run is recorded).

## What works today

- Upload a CSV file through the Streamlit interface.
- Input validation: empty files, non-UTF-8 files, malformed CSV, oversized datasets.
- Dataset profiling computed with Pandas: shape, data types, missing values, unique counts, most common values, duplicate rows, numeric summary statistics.
- Automatic data-quality warnings (columns with heavy missingness, constant columns, unnamed columns).
- Six deterministic analytical tools (see below) exposed through a tool registry.
- **An autonomous investigator** (`src/investigator.py`) that turns a natural-language question into a validated plan of tool calls, executes them in Python and reports evidence-backed findings.
- A 15-question evaluation set with hand-checked expected answers and machine-checkable answer checks (`eval/questions.json`).
- An **evaluation harness** (`src/evaluation.py` + `eval/score_investigator.py`) that measures tool accuracy, parameter accuracy, refusal accuracy, answer accuracy and faithfulness, and records latency, tokens and (optionally) cost.
- **Safety guardrails**: the model can only name registered tools and declared parameters; plans are grounded to real columns; `redact_columns` keeps chosen fields (e.g. identifiers) out of every prompt; and a check flags statistics in the narrative that no tool computed.
- **The investigator in the browser**: upload a CSV (or load the sample dataset), ask a question, and get a report that keeps computed findings separate from the AI answer. Example questions are derived from the real schema, PII columns are pre-selected for redaction, and the UI has a loading spinner, error states and reset/clear.
- **Charts from computed results**: `src/visualizer.py` builds bar, line and heatmap charts from tool result tables (altair), rendered in the report view. No chart is ever generated from model text.
- **Safe activity logging**: every investigation is appended as one JSON line with metadata only (question, shape, tools, params, status, tokens, latency) — never cell values or the API key.
- **Security review** (`docs/security.md`): secrets handling (no keys in the repo), input validation and sanitisation, rate limits and caching, a security checklist, a triaged bug list with fixes, and remaining risks.
- An isolated Gemini Flash client (`src/llm_client.py`) using plain HTTP, with typed errors for authentication, rate limits, timeouts and malformed responses, plus retries on transient overload.
- Unit tests for the profiling, tools, LLM client, investigator, evaluation, safety, presentation and interface modules (`pytest`, 271 tests). Every LLM-dependent test runs offline against a fake model.

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

`TOOL_REGISTRY` describes each tool and its parameters in a machine-readable form, and `run_tool(name, df, **params)` executes one by name. This is the interface the investigator hands to the language model, and it is also why the tools are separate from the UI and the LLM.

## The investigator

`src/investigator.py` is the agent layer. It takes a question and a DataFrame and produces an `InvestigationReport`. The pipeline has one seam — an object exposing `generate_json(...)` — which is the real Gemini client in production and a fake in the tests.

```
question + dataset + profile
        │
        ▼
 build_dataset_context()   schema, types, missing %, uniques, category values, date-like flags
 build_tool_catalogue()    built from TOOL_REGISTRY, so the registry stays the single source of truth
        │
        ▼  Gemini (structured JSON output, MEDIUM thinking → LOW fallback on timeout)
 validate_plan()           strict: known tool, known params, required present, types, enums,
        │                  and every column name checked against the real dataset
        ▼
 run_plan()                run_tool() per step; a failing step is recorded, not raised
        │
        ▼
 build_evidence()          deterministic Markdown tables — the ground truth
        │
        ▼  Gemini (structured JSON output)
 InvestigationReport       computed findings and AI interpretation kept in separate fields
```

**Guarantees**

- The model only ever chooses a tool name and parameters. It cannot run code, and `run_tool` is the only executor.
- The model never sees the dataset itself — only its schema and profile. Prompts are capped in size and contain no whole columns.
- A plan that names an unknown tool, an unknown parameter, a missing required parameter, a wrong type, or a column that does not exist is **rejected before anything runs** (`PlanError`).
- Structured output is **normalised before the strict check**: the model occasionally appends its own reasoning text to the end of a string field (e.g. `"revenue metric: mean"`), so a value that starts with a known column or enum value is snapped back to it. Values that do not start with a known token are still rejected, and every normalisation is shown in the report under "Normalised parameters".
- Two transient model failures are retried once: a **timeout** (retried at the lower thinking level, which is faster) and a **malformed JSON body** (a sampling glitch, retried once before giving up).
- **Sensitive columns can be redacted.** Pass `redact_columns=[...]` and those columns are still described by name and type, but no value from them reaches the planning prompt, the schema block, the evidence tables or the interpretation prompt.
- `investigate()` never raises on expected failures. It returns a report whose `status` is `ok`, `partial` (findings computed but interpretation failed), `unsupported` or `error`, so a UI can always render something honest.
- The report separates **Findings (computed in Python)** from the **Answer (AI interpretation)** in distinct sections, and flags any statistical-looking number in the narrative that cannot be matched to a computed value.

### Run the evaluation

The scoring script runs the planning stage against every question in `eval/questions.json` and scores four independent things from the run's own computed findings: **tool accuracy**, **parameter accuracy**, **refusal accuracy** and **answer accuracy**. Because the tools are deterministic, answer accuracy can be measured from a plan-only run, so the whole suite costs one model call per question. It also records latency and token usage.

```powershell
.\.venv\Scripts\python.exe eval\score_investigator.py
```

| Flag | Purpose |
|---|---|
| `--model <name>` | Score a different model. |
| `--out <path>` | Where to write the Markdown report. |
| `--interpret N` | Also run the narrative stage on the first N questions, to measure **faithfulness**. |
| `--pause <seconds>` | Delay between questions (default 12, for the free-tier rate limit). |
| `--price-in` / `--price-out` | USD per 1M tokens, for a paid-tier cost estimate. |

Exit codes: `0` all passed, `2` some failed, `3` the run stopped early on a rate limit. The results, the failure analysis and the fixes are written up in `eval/evaluation.md`.

> **A full investigation costs two requests** (one to plan, one to interpret). On the free tier `gemini-3.6-flash` allows 20 requests per day, so roughly ten complete investigations per day. The scorer pauses between questions to respect the per-minute limit, and treats a daily-quota 429 as a stopped run rather than as a set of failures.

## Planned features

- **Visualisations:** charts built from actual computed results.
- **Structured report in the app:** wire the investigator into the Streamlit interface so a question typed in the browser returns a full report.

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
| Gemini Flash (REST) | Question interpretation and planning | In use |

## Project structure

```
data-detective/
├── app.py                  # Streamlit interface (upload + profile view)
├── src/
│   ├── __init__.py         # Marks src/ as a Python package
│   ├── profiler.py         # CSV validation + profiling (no UI, no LLM)
│   ├── tools.py            # Predefined analytical tools + registry
│   ├── llm_client.py       # Isolated Gemini REST client (the only LLM code)
│   ├── investigator.py     # Agent: plan -> validate -> execute -> evidence report
│   ├── evaluation.py       # Offline scoring: metrics, answer checks, usage recording
│   ├── ui_report.py        # Presentation helpers turning a report into UI data
│   ├── audit.py            # Safe, metadata-only investigation logging
│   └── visualizer.py       # Charts built from computed tool results (altair)
├── tests/
│   ├── test_profiler.py    # Unit tests for src/profiler.py
│   ├── test_tools.py       # Unit tests for src/tools.py
│   ├── test_eval_set.py    # Guards eval/questions.json against drift
│   ├── test_llm_client.py  # Unit tests for src/llm_client.py (no network)
│   ├── test_investigator.py# Unit tests for src/investigator.py (no network)
│   ├── test_evaluation.py  # Unit tests for src/evaluation.py (no network)
│   ├── test_safety.py      # Sandbox, grounding, redaction and honesty guardrails
│   ├── test_ui_report.py   # Presentation helpers (no Streamlit, no network)
│   ├── test_audit.py       # Safe logging (no network)
│   ├── test_app.py         # Streamlit interface smoke tests (AppTest, no browser)
│   ├── test_visualizer.py  # Chart builders (no network)
│   └── test_input_validation.py  # Input validation and sanitisation
├── eval/
│   ├── questions.json      # Fixed question set with expected answers + answer checks
│   ├── score_investigator.py  # Runs the question set and writes a metrics report
│   ├── evaluation.md       # Day 4: approach, metrics, failure analysis, fixes
│   ├── evaluation-gemini-3.1-flash-lite.md  # Full backup-model run (12/15)
│   ├── baseline.md         # Day 3 primary-model baseline, diagnosis and fix
│   ├── baseline-gemini-3.6-flash-prefix.md  # Raw pre-fix 9/15 run (for the record)
│   └── baseline-gemini-3.1-flash-lite.md  # Backup-model baseline (11/15)
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
>
> **Free-tier daily quota.** `gemini-3.6-flash` allows **20 requests per day** on the free tier, and a full investigation uses two (plan + interpret). A full 15-question scoring run needs more than one day's budget, so the baseline is built up across runs. The client retries transient `429`/`5xx` responses with backoff and reports a daily-quota `429` as a clear `LLMRateLimitError` rather than a crash.

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

## Security and data safety

- Do not upload confidential or personally identifiable information during development.
- Validate uploaded files before processing.
- Only the dataset's schema and profile are sent to the model, never whole columns.
- Sensitive columns can be redacted with `redact_columns` so no value from them reaches the model in any prompt.
- Do not permanently store user datasets.
- Use public datasets in accordance with their individual licences.
- Never commit datasets or API credentials.
- A full security review, checklist and triaged bug list live in `docs/security.md`.

## Roadmap

- [x] Environment, repository and dependency pinning
- [x] CSV loading and dataset profiling with tests
- [x] Streamlit upload and profile interface
- [x] Predefined analytical tools (comparison, correlation, trend, outliers) with tests
- [x] Fixed evaluation question set with expected answers
- [x] Isolated Gemini REST client with typed error handling and tests
- [x] LLM planning with structured tool selection
- [x] Evidence validation and investigation report
- [x] Evaluation harness with tool, parameter, refusal and answer accuracy
- [x] Latency, token and cost measurement
- [x] Safety guardrails: sandbox, column grounding, PII redaction, invented-number check
- [x] Prompt iteration driven by evaluation failures
- [x] Investigator wired into the Streamlit interface (loading, errors, reset, safe logging)
- [x] Security hardening: input validation, secrets review, rate limits and caching, bug triage
- [x] Visualisations: charts built from computed results (altair)
- [x] Deployment live: https://data-detective-r.streamlit.app/ (Streamlit Community Cloud)

## Deployment

The app is deployment-ready but the public URL needs an interactive Streamlit Cloud
sign-in. Full steps, a smoke-test checklist, performance notes and a fallback plan
live in [`docs/deployment.md`](docs/deployment.md).

- **Live app:** https://data-detective-r.streamlit.app/ (Streamlit Community Cloud).
- **Local run:** `streamlit run app.py` (opens http://localhost:8501).
- **Container:** `docker build -t data-detective . && docker run -p 8501:8501 -e GEMINI_API_KEY=... data-detective`.
- **Secrets:** set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) in the platform's secrets; `.streamlit/secrets.toml` is git-ignored.

## Challenge

Developed as part of the **Decoding Data Science (DDS) Academy AI Application Building Challenge — November 2026**.
