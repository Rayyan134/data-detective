# Day 6 LinkedIn post — Security & debugging

Screenshot to attach: `docs/screenshots/day6-security.png`

---

**Draft post**

Day 6 of the DDS AI Application Building Challenge: no new features — a hardening pass. 🛡️

"Data Detective" lets an LLM plan analyses over your CSV. That means the attack surface is real, so today I went through it deliberately.

Security review (full write-up in `docs/security.md`):
🔒 No secrets in the repo — `git grep` for key-shaped strings returns only the `replace-me` placeholder in `.env.example`.
🔒 Inputs are bounded — CSV size/row/column limits, a 2000-char question cap, filenames reduced to their basename (never used as a path).
🔒 The model can only name tools and parameters from the fixed registry; `validate_plan` rejects anything else *before* it runs.
🔒 Prompt injection in the question cannot widen the plan — it still comes back as a clean error report.
🔒 PII redaction is opt-in and value-free: no value from a redacted column reaches any prompt or evidence table.
🔒 The activity log contains metadata only — no cell values, no API key (verified by test).

Bug list triaged (6 items), including the ones that actually bit me:
1. Leaked reasoning text in a parameter value (Day 3) → normalised back to a known token.
2. Silent mean default when `metric` was omitted (Day 4) → prompt now demands an explicit statistic.
3. No refusal on out-of-scope questions (Day 4) → prompt forbids column substitution; verified live.
4. Malformed JSON bodies (Day 4) → retry once.
5. Over-strict test expectation on Q07 → relaxed and documented.
6. Interface had no automated coverage → AppTest smoke suite.

Remaining risks, stated honestly: free-tier quota (20 req/day/model) still blocks the authoritative primary-model run; p95 latency ~15s with only a spinner; no charts yet; public deployment is Day 7; the honesty check is best-effort string matching; redaction is opt-in.

Also: a per-session investigation cache so re-running an identical question makes no API call, plus cached file loading. 274 tests pass offline.

Code: https://github.com/Rayyan134/data-detective

#AI #LLM #AgenticAI #Security #DataScience #BuildInPublic #DecodingDataScience

---

## Notes for the author

- Screenshot is the Day 6 security review card (`docs/screenshots/day6-security.html` → PNG).
- Security details live in `docs/security.md`.
- The public demo link is still pending (Day 7).
