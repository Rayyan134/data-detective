# Security review (Day 6)

## Secrets handling

- `.env` is git-ignored; only `.env.example` (with `replace-me` placeholders) is committed.
- `git ls-files` contains no `.env` and no key-shaped string (`git grep` for `AIza…` / `sk-…` returns nothing except the placeholder in `.env.example`).
- The API key is read from the environment inside `src/llm_client.py` and is never written into code, error messages or logs.
- Streamlit secrets (`.streamlit/secrets.toml`) are git-ignored.

## Input validation and sanitisation

- CSVs go through `src/profiler.load_csv`: UTF-8 (with BOM), size/row/column limits, parse errors, empty files, malformed CSV — all raise `DatasetError` with a readable message.
- The question is capped at `MAX_QUESTION_LENGTH` (2000 chars) in `src/investigator.investigate`; empty questions are rejected.
- Uploaded filenames are reduced to their basename and capped at 120 chars before being used as a label; they are never used as a path.
- Redacted columns are validated against the real schema; an unknown name is ignored rather than crashing.
- The model can only name tools and parameters from the fixed registry; `validate_plan` rejects anything else before it runs.

## Rate limits and caching

- `src/llm_client` retries transient `429`/`5xx` with backoff (honours `Retry-After`, capped at 8s) and surfaces a daily-quota `429` as `LLMRateLimitError`.
- The scorer treats a daily-quota `429` as a stopped run (exit code 3), not as question failures.
- The Streamlit app caches investigations in `st.session_state` keyed by (question, dataset, redactions), so re-running an identical question makes no API call.
- `load_uploaded_csv` is cached by file bytes (`st.cache_data`).
- Free-tier quota is 20 requests/day/model; a full investigation costs 2, so roughly 10 complete investigations per day.

## Security checklist

- [x] No secrets in the repository (verified by `git grep`).
- [x] `.env` ignored; `.env.example` committed with placeholders.
- [x] User input (CSV, question, filename) validated and bounded.
- [x] Model output validated before execution (tool, params, columns).
- [x] Prompt injection in the question cannot widen the plan (rejected by `validate_plan`).
- [x] PII redaction is opt-in and value-free (no value from a redacted column reaches any prompt or evidence table).
- [x] Activity log contains metadata only (no cell values, no API key).
- [ ] Rate-limit handling is per-process only; a multi-instance deployment would need a shared limiter (out of scope for the demo).

## Top bugs fixed

1. **Leaked reasoning text in a parameter** (Day 3): the model appended its own reasoning to a string field (`"revenue metric: mean"`). Fixed with a normalisation step that snaps a value back to a known token when it starts with one; values that do not match are still rejected. Verified live.
2. **Silent mean default** (Day 4): omitting `metric` defaulted to the mean, so "total revenue" and "median revenue" were answered with averages. Fixed by demanding an explicit statistic in the planning prompt. Not yet live-verified (quota).
3. **No refusal on out-of-scope questions** (Day 4): "customer segment" was mapped onto an unrelated column. Fixed by forbidding column substitution; verified live — Q15 now refuses with "The column 'customer segment' is missing from the dataset."
4. **Malformed JSON bodies** (Day 4): a sampling glitch, not an answer. Fixed by retrying once on `LLMResponseError`.
5. **Over-strict test expectation** (Day 4): Q07's `top_n: 3` did not change the answer with three products. Relaxed the expectation and documented why `top_n` is not asserted.
6. **AppTest smoke coverage** (Day 5): the Streamlit interface is now exercised end-to-end (empty state, sample load, full investigation, failure surfacing) without a browser or network.

## Remaining risks

- Free-tier quota (20 requests/day/model) means the authoritative primary-model evaluation run is still pending.
- Latency is long-tailed (p95 ~15s); the UI only shows a spinner, with no cancellation or per-step progress.
- Visualisations are not implemented yet.
- Public deployment (Day 7) is outstanding; the demo is local-only today.
- The honesty check (`find_unverified_numbers`) is best-effort string matching and can miss a restated number; flagged numbers are surfaced, never presented as evidence.
- Redaction is opt-in; a user who does not select a sensitive column will send its values to the model.
