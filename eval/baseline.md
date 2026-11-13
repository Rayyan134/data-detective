# Investigator baseline

- Date: 2026-11-13
- Dataset: `data/sample_sales.csv`
- Mode: planning only — the chosen tool and its parameters are scored, the narrative interpretation is not
- Scorer: `eval/score_investigator.py`

This file records the honest state of the baseline. A clean, single-run score
for the primary model (`gemini-3.6-flash`) could not be completed on the day
Stage C landed, because that model's free tier allows only **20 requests per
day** and the quota was exhausted while debugging. The evidence below is what
was actually observed, in the order it happened.

## Primary model: `gemini-3.6-flash`

**Status: clean re-run pending quota reset.**

### Run 1 — pre-fix, 6 / 15

The first full scoring run scored **6 / 15**. It failed for three separate
reasons, only one of which was a model limitation:

| Cause | Questions | Explanation |
|---|---|---|
| Garbled parameter values | Q01, Q02, Q11 and others | See "Root cause" below. |
| Scorer compared omitted optional parameters literally | Q02, Q07 | The model correctly omitted an optional parameter (e.g. `metric`), which means "use the default"; the scorer treated the omission as a mismatch. Fixed in `_params_match`. |
| Free-tier rate limit (HTTP 429) | a few | Retried on the next question. |

### Root cause of the garbled values

With `thinking_level="MEDIUM"`, thinking tokens count towards
`maxOutputTokens`. The original cap (`2048`) was too tight for the JSON plan,
so the model ran out of budget mid-string and emitted degraded text inside
parameter values — e.g. a `value_column` of
`"revenue Brian? no, revenue! wait, ..."` instead of `"revenue"`. Validation
correctly rejected those plans, so the tools were never run on bad input.

**Fix:** raised the caps to `PLAN_MAX_OUTPUT_TOKENS = 8192` and
`INTERPRETATION_MAX_OUTPUT_TOKENS = 4096`, leaving room for reasoning plus the
answer. The second failure was a scorer bug and was fixed separately.

### Run 2 — post-fix verification (partial, quota exhausted)

After the fix, two spot checks were re-run against the live API:

- **Q11** ("Which month had the highest total revenue?") → plan
  `trend_over_time {"date_column": "order_date", "freq": "ME", "value_column": "revenue"}`
  — **validated, correct.** This is the same question that was garbled before
  the fix, so the token cap was confirmed as the cause.
- **Q01** ("Which region has the highest average revenue per order?") →
  `group_comparison {"group_column": "region", "metric": "mean", "value_column": "revenue"}`
  — **validated, correct**, answer East with mean 912.007.

The full 15-question re-run then hit the **daily** limit:

```
Quota exceeded for metric:
generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 20, model: gemini-3.6-flash,
description: Generate content requests per day limit exceeded on Free Tier.
```

A full investigation costs two requests (plan + interpret), so the 20-request
daily budget allows only about ten complete investigations per day. The clean
`gemini-3.6-flash` baseline will be recorded here once the quota resets, by
running:

```
.\.venv\Scripts\python.exe eval\score_investigator.py
```

## Backup model: `gemini-3.1-flash-lite` — 11 / 15

Because the primary model was blocked, the same exam was run against
`gemini-3.1-flash-lite`, which still had quota. Full detail is in
`eval/baseline-gemini-3.1-flash-lite.md`.

- **Score: 11 / 15.**
- Failures: **Q02** and **Q13** (a weaker model omitted or chose the wrong
  `metric`), **Q07** (wrong `top_n`), and **Q15** (did not return an explicit
  refusal for an unsupported question).
- The lite model is less capable than the primary model, so this is a
  conservative lower bound on the investigator's planning quality, not a
  substitute for the primary baseline.

## What the baseline measures

Only the planning stage is scored: the chosen tool must match the hand-checked
expectation, and every expected parameter must match (an omitted optional
parameter is compared against the tool's declared default). Tool selection is
what the 15-question exam was designed to test, and it is the part of the
pipeline the model is responsible for. Execution and reporting are covered
offline by `tests/test_investigator.py`.
