# Investigator baseline

- Date: 2026-11-13 (initial), revised 2026-11-15
- Dataset: `data/sample_sales.csv`
- Mode: planning only — the chosen tool and its parameters are scored, the narrative interpretation is not
- Scorer: `eval/score_investigator.py`

## Primary model: `gemini-3.6-flash` — 9 / 15 (pre-normalisation)

The first clean full run of the primary model scored **9 / 15**. The raw table
is in `eval/baseline-gemini-3.6-flash-prefix.md`. Every one of the six failures
was the **same defect**, not six different mistakes:

| ID | Question | Failure |
|---|---|---|
| Q01 | Which region has the highest average revenue per order? | leaked text in `value_column` |
| Q02 | Compare total revenue by product. | leaked text in `value_column` |
| Q04 | How did monthly revenue change over time? | leaked text in `value_column` |
| Q10 | What is the average satisfaction score by product? | leaked text in `value_column` |
| Q11 | Which month had the highest total revenue? | leaked text in `value_column` |
| Q13 | Which channel has the highest median revenue per order? | leaked text in `value_column` |

Q03, Q05, Q06, Q07, Q08, Q09, Q12, Q14 and Q15 passed.

### The defect

In each failure the model wrote a *correct* column name and then kept
"thinking" inside the same JSON string, for example:

```
"value_column": "revenue oppose metric sum"
"value_column": "revenue নদী_freq="
"value_column": "revenue Cholera freq ME"
```

Strict validation correctly rejected these plans, so no tool was ever run on a
bad column name. The consequence was a failed question, not a wrong answer.

### Corrected root cause

An earlier note in this file blamed the `maxOutputTokens` cap. That was wrong.
A targeted probe on 2026-11-15 showed:

- `finish_reason: STOP` — generation was **not** truncated (`MAX_TOKENS` would
  have indicated truncation).
- `thoughtsTokenCount: 297` — thinking used very little budget.
- The leak reproduced at **both** `MEDIUM` and `LOW` thinking.

So this is a structured-output quirk of the model: the reasoning text bleeds
into the end of a string field, independently of the token budget.

### The fix

`validate_plan` now **normalises** a parameter value before the strict check:

- If a string parameter that must name a column (or must be one of an enum)
  does not match, but a known column name (or enum value) is a prefix of it, the
  value is snapped back to the known token.
- A value that does not start with any known token is still rejected exactly as
  before, so a genuinely wrong column or enum is never accepted.
- Every normalisation is recorded on the plan and shown in the report under
  **"Normalised parameters"**, so the change is visible rather than hidden.

This is a deterministic, offline-testable fix (`tests/test_investigator.py`),
not a rewrite of the model's intent: it only trims trailing noise after an
unambiguously correct value.

### Verification of the fix (live)

Two of the six failing questions were re-run live on 2026-11-15:

- **Q01** → `group_comparison {"group_column": "region", "metric": "mean", "value_column": "revenue"}`
  — clean, no repair needed.
- **Q04** → leaked `'revenue Cholera freq ME'`, normalised to `'revenue'`,
  plan validated, status `ok`.

### Pending: full re-run

The free tier allows only **20 requests per day** for `gemini-3.6-flash`, and
all 20 were used on 2026-11-15 (one full 15-question run plus diagnostics).
The full post-fix score will be recorded here by re-running:

```
.\.venv\Scripts\python.exe eval\score_investigator.py
```

The pre-fix raw output is kept as `eval/baseline-gemini-3.6-flash-prefix.md`.

## Backup model: `gemini-3.1-flash-lite` — 11 / 15

Run against the same exam while the primary model was quota-blocked. Full
detail is in `eval/baseline-gemini-3.1-flash-lite.md`.

- **Score: 11 / 15.**
- Failures: **Q02** and **Q13** (omitted or chose the wrong `metric`), **Q07**
  (wrong `top_n`) and **Q15** (no explicit refusal for an unsupported question).
- This run predates the normalisation fix, and the lite model is less capable
  than the primary model, so treat it as a conservative lower bound.

## What the baseline measures

Only the planning stage is scored: the chosen tool must match the hand-checked
expectation, and every expected parameter must match (an omitted optional
parameter is compared against the tool's declared default). Tool selection is
what the 15-question exam was designed to test, and it is the part of the
pipeline the model is responsible for. Execution and reporting are covered
offline by `tests/test_investigator.py`.
