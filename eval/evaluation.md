# Investigator evaluation (Day 4)

- Date: 2026-11-15
- Dataset: `data/sample_sales.csv`
- Question set: `eval/questions.json` (15 questions)
- Harness: `src/evaluation.py` + `eval/score_investigator.py`
- Offline tests: `tests/test_evaluation.py`, `tests/test_eval_set.py`, `tests/test_safety.py`

This document describes **how** the investigator is evaluated, **what** the
current numbers are, and **what was changed** in response to the failures. The
machine-written run output lives beside it as
`eval/evaluation-gemini-3.1-flash-lite.md`.

## 1. Evaluation approach

The Day 3 baseline could only ask one question — *did the model choose the
expected tool?* — because the scorer threw the computed tables away. Day 4 keeps
them and measures five independent things from a single pass.

| Metric | What it asks | Denominator |
|---|---|---|
| Tool accuracy | Did the plan call the expected tool? | all 15 questions |
| Parameter accuracy | Did it pass the expected arguments? | the 13 supported questions |
| Refusal accuracy | Did it decline the 2 questions no tool can answer? | the 2 unsupported questions |
| Answer accuracy | Does the **computed evidence** contain the expected answer? | all 15 questions |
| Plan accuracy | Right tool + right parameters, or a correct refusal | all 15 questions |
| Faithfulness | Did the narrative avoid numbers no tool computed? | the questions that were interpreted |

Each accuracy has its own denominator on purpose: a "not applicable" is never
counted as a pass or a fail. `refusal_accuracy` is `n/a` when no refusal
question was scored, and `parameter_accuracy` is `n/a` when nothing was
supported.

### Why answer accuracy is measured from the evidence

The tools are deterministic, so the tools still run in a **plan-only** run (the
only thing skipped is the narrative call). That means one model call per
question is enough to score four of the five metrics, and answer accuracy is
checked against the values the tools actually produced.

This is deliberately a *ceiling*, not a claim that the agent's chosen statistic
is right. If the model computes `mean` where the question asked for `sum`, the
evidence still contains the right product total somewhere in the table, so
answer accuracy passes — and **parameter accuracy is what catches the mistake**.
Keeping the two separate is what makes the mean/median/sum confusion (the most
common real failure) visible instead of hiding it.

### The question set

15 hand-written questions over `data/sample_sales.csv`, each carrying:

- `expected_tool` (or `null` for the two out-of-scope questions),
- `expected_params` (compared against the tool's declared defaults, so omitting
  an optional parameter is correct),
- a machine-checkable `check` describing the expected answer (`top_row`, `cell`,
  `matrix_cell`, `cell_value`, `row_count`, `refuse`),
- a hand-computed `expected_answer_hint` so a human can verify the check.

The set is designed to test judgement, not just plumbing: it includes a
mean-vs-median trap (Q13), a weak correlation that must not be over-claimed
(Q03), a near-mechanical correlation that must not be treated as an insight
(Q12), a real-but-tiny difference (Q10), and two questions that **must be
refused** because no available tool can answer them (Q14, Q15).

`tests/test_evaluation.py` asserts the set is self-consistent: the expected plan
for every question actually passes that question's own check on the real data.

## 2. The harness

`src/evaluation.py` is pure Python over an `InvestigationReport`; it never calls
the model, so it is fully covered offline. `eval/score_investigator.py` drives
it:

```
.\.venv\Scripts\python.exe eval\score_investigator.py --out eval\evaluation-<model>.md --interpret 3
```

| Flag | Purpose |
|---|---|
| `--model` | Override the model (defaults to `GEMINI_MODEL` / the client default). |
| `--out` | Where to write the Markdown report. |
| `--interpret N` | Also run the narrative stage on the first N questions (faithfulness). |
| `--pause` | Seconds between questions (default 12, for the free-tier rate limit). |
| `--price-in` / `--price-out` | USD per 1M tokens, for a paid-tier cost estimate. |

Exit codes: `0` all questions passed, `2` some failed, `3` the run stopped early
on a rate limit.

**Rate-limit handling.** A free-tier HTTP 429 is a quota event, not a model
miss. The scorer detects it, stops the run, and lists the unattempted questions
under *Not attempted* instead of recording them as failures. (The first
attempted primary-model run was thrown away for exactly this reason.)

Latency and token usage are recorded at the client seam by
`evaluation.RecordingClient`, which wraps the real client. A retried call
records both attempts, so the latency figure includes the wait the user felt.

## 3. Results

### Primary model `gemini-3.6-flash` — pending

The free tier allows **20 requests per day, per project, per model**, and those
were spent during the Day 3 work and diagnostics. The authoritative primary-model
run will be recorded as `eval/evaluation-gemini-3.6-flash.md` once the quota
resets:

```
.\.venv\Scripts\python.exe eval\score_investigator.py --out eval\evaluation-gemini-3.6-flash.md --interpret 3
```

A partial primary-model run was captured before the quota was exhausted and then
deleted, because a mostly-rate-limited run is not a baseline. Its two useful
signals were kept: the primary model is **slower** (roughly 9–12 s per question,
versus 5.7 s mean for the lite model) and it spends **thinking tokens**
(~1,185 over 4 calls) where the lite model spends none. The two genuine answers
it produced matched the lite model's behaviour (one clean plan, one `metric`
omission).

### Backup model `gemini-3.1-flash-lite` — full run

A full 15-question run on the backup model, which has its own separate daily
quota. Full detail: `eval/evaluation-gemini-3.1-flash-lite.md`.

| Metric | Value |
|---|---|
| Plan accuracy | 11/15 = 73% (12/15 = 80% after correcting Q07, see below) |
| Tool accuracy | 14/15 = 93% |
| Parameter accuracy | 10/13 = 77% (11/13 = 85% after correcting Q07) |
| Refusal accuracy | 1/2 = 50% |
| Answer accuracy | 14/15 = 93% |
| Faithfulness | 3/3 interpreted questions had no unverified numbers |

Four questions did not pass. **Three were genuine** model errors; **one was a
mistake in the test set**:

| ID | What happened | Real defect? |
|---|---|---|
| Q02 | "Compare total revenue by product" → chose `mean`, not `sum`. | Yes — metric omission |
| Q13 | "highest **median** revenue" → chose `mean`, not `median`. | Yes — metric omission |
| Q15 | "customer segment" mapped onto `channel` instead of refusing. | Yes — no refusal |
| Q07 | Chose `top_n: 3` for three products. | **No** — over-strict expectation |

Q07 was the test set's fault: `top_n` only caps how many categories are shown
and does not change the answer when there are three products. The expectation
was relaxed to `{"column": "product"}`, which makes the recorded run 12/15.

## 4. Failure modes and the fixes applied

All three real failures were **planning-prompt gaps**, not tool or validation
bugs: the tools were right, the model just described the request loosely.

**Fix 1 — demand an explicit statistic.** `PLAN_SYSTEM_INSTRUCTION` now requires
`metric` to be set whenever the question implies one, with a mapping ("total",
"sum" → `sum`; "average", "per order" → `mean`; "median", "typical" → `median`).
This targets Q02 and Q13.

**Fix 2 — forbid substituting a different column.** The same instruction now
says that if the question names something the schema does not contain (a
customer, a segment, a margin), the plan must be `unsupported` and name the
missing column — never mapped onto a lookalike. This targets Q15.

**Verified live** (backup model, 2026-11-15):

- Q15 now correctly refuses: *"The column 'customer segment' is missing from the
  dataset."*
- Q13 hit a *different*, unrelated failure in the same check: the model returned
  malformed JSON (`Unterminated string starting at: line 9 column 25`).

**Fix 3 — retry a malformed response.** A malformed body is a sampling glitch,
not an answer, so `_generate_json` now retries once on `LLMResponseError`
(preferring the lower thinking level when one is configured) before giving up.
A timeout still retries at the fallback thinking level as before. Covered by
`tests/test_investigator.py`.

**Fix 4 — correct the test set.** The over-strict Q07 expectation was relaxed
(see above), and the notes for Q06/Q07 now record why `top_n` is not asserted.

Every fix is offline-testable and has a test that fails without it.

## 5. Latency and cost

| Signal | Backup model `gemini-3.1-flash-lite` |
|---|---|
| Latency mean | 5.72 s |
| Latency p50 | 4.08 s |
| Latency p95 | 15.54 s |
| Latency max | 19.54 s |
| Prompt tokens (15 questions) | 22,417 |
| Output tokens | 1,869 |
| Thinking tokens | 0 |
| Total tokens | 24,286 (≈1,619 per question) |

Notes:

- Latency is **long-tailed**: the median question is ~4 s but the p95 is ~15 s,
  so any UI needs a visible loading state and a timeout well above 20 s.
- The run is **prompt-heavy** (prompt tokens are ~92% of the total): every
  question resends the tool catalogue and the schema block. The obvious
  optimisation, if it ever matters, is to cache or trim that prefix.
- On the free tier the marginal cost is **$0**. A paid-tier estimate can be
  produced without changing any code by setting `--price-in` / `--price-out`
  (or `GEMINI_PRICE_INPUT_PER_MTOK` / `GEMINI_PRICE_OUTPUT_PER_MTOK`); the
  harness refuses to guess a price when none is supplied, and reports "free
  tier" instead.
- Free-tier quota is **20 requests/day/model**, which is why the whole harness
  runs plan-only by default: a full evaluation is 15 requests, leaving room for
  a small faithfulness pass.

## 6. Safety and guardrails

The investigator is built so that the model **plans** but never **executes**.
`tests/test_safety.py` (22 tests) pins the four properties that matter.

**Sandbox.** The model may only name a tool from the fixed catalogue, and only
that tool's declared parameters. A plan naming `os.system`, `subprocess.run`,
`eval` or `__import__` is rejected by `validate_plan` before anything runs; an
undeclared parameter (a SQL fragment, a path) is rejected too; a `"code"` key
smuggled next to the real fields is ignored. A prompt-injection attempt in the
*question* cannot widen the plan — it still comes back as a clean error report
with no steps executed. `run_tool` independently rejects an unregistered name,
and a test asserts the registry contains only the six analysis functions.

**Grounding.** A plan may only reference columns that exist. A clearly
misspelled column is snapped back to the real one and the repair is recorded on
the plan and shown in the report; a genuinely absent column is rejected.

**Privacy / PII.** `redact_columns` threads through `build_dataset_context`,
`build_plan_prompt`, `build_interpretation_prompt`, `build_evidence`,
`plan_investigation` and `investigate`. A redacted column is still described by
name and type (so the schema stays truthful and the model can reason about it),
but **no value from it** appears in the planning prompt, the schema block, the
evidence tables or the interpretation prompt. `find_outliers` returns whole
rows, so this is what keeps an identifier column such as `order_id` out of the
request even when a tool would otherwise include it. Redaction removes the
identifier, never the answer: numeric findings are untouched.

**Honesty.** `find_unverified_numbers` flags statistics in the narrative that
cannot be matched to a computed value (allowing for rounding, sign and
percentage/fraction restatement) and ignores small counts. Flagged numbers are
surfaced in the report under a caution heading, never presented as evidence.

## 7. Reproducing

```
# offline: 232 tests, no API key needed
.\.venv\Scripts\python.exe -m pytest -q

# live: full evaluation on the primary model (needs GEMINI_API_KEY)
.\.venv\Scripts\python.exe eval\score_investigator.py --out eval\evaluation-gemini-3.6-flash.md --interpret 3
```
