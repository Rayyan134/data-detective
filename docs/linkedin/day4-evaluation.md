# Day 4 LinkedIn post — Application evaluation

Screenshot to attach: `docs/screenshots/day4-evaluation.png`

---

**Draft post**

Day 4 of the DDS AI Application Building Challenge: I stopped trusting my agent and started measuring it. 🔍

"Data Detective" takes a CSV and a plain-English question and returns an evidence-backed answer. The model only *plans* — it picks a tool and its arguments. Every number is computed in Python.

Today I built the exam.

A fixed set of 15 questions over a synthetic sales dataset, each with a hand-checked answer, scored automatically by a harness that reports five metrics — each on its own honest denominator:

📊 Plan accuracy 80% (12/15)
🧰 Tool accuracy 93%
🎛️ Parameter accuracy 85%
✅ Answer accuracy 93%
🕵️ Faithfulness 3/3 — no invented numbers

Then I fixed what the exam exposed. All three real failures were *prompt* gaps, not code bugs:

1️⃣ "Total revenue by product" came back as a mean. The planner now must name the statistic it is computing (total → sum, typical → median).
2️⃣ "Customer segment" was quietly mapped onto an unrelated column. The planner must now refuse a question the data cannot answer. Verified live — it now says: "The column 'customer segment' is missing from the dataset."
3️⃣ Occasional malformed JSON. The client now retries once before giving up.

I also hardened it: a tool sandbox (the model can never run code), column grounding, opt-in PII redaction, and an honesty check that flags any number in the narrative that no tool actually computed.

232 offline tests, no API key needed.

Two honest notes:
• The run above is on the backup model — the free tier's 20 requests/day/model quota for the primary model is exhausted. The authoritative run is queued for when it resets.
• Latency is long-tailed (p95 ≈ 15s), so the UI will need a real loading state.

Code: https://github.com/Rayyan134/data-detective

#AI #LLM #AgenticAI #Evaluation #DataScience #BuildInPublic #DecodingDataScience

---

## Notes for the author

- The screenshot is the evaluation summary card (`docs/screenshots/day4-evaluation.html` → PNG).
- Numbers are from `eval/evaluation.md` §3, on the backup model `gemini-3.1-flash-lite`, with the Q07 test-set correction applied (12/15 plan, 85% parameters).
- The primary-model run (`gemini-3.6-flash`) is blocked by the free-tier daily quota and is deliberately labelled as pending rather than replaced with a partial run.
