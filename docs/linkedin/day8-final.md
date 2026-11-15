# Day 8 LinkedIn post — Final submission

Screenshot to attach: `docs/screenshots/day5-interface.png` (or any of the Day 4/6/7 cards)

---

**Draft post**

8 days. One AI agent that investigates your data — and never invents a number. 🎯

"Data Detective" is my submission for the DDS AI Application Building Challenge (Nov 2026). You upload a CSV, ask a question in plain English, and get back computed findings (real tables, real numbers) kept strictly separate from an AI narrative that is explicitly labelled as interpretation.

The agent never runs code and never sees your data — only its schema and profile. The LLM only *plans*: it picks a tool and its arguments from a fixed registry. Python computes everything. A validator rejects any plan that names an unknown tool, parameter or column before anything runs.

By the numbers:
• 6 deterministic analytical tools (comparison, correlation, trend, outliers, value counts, missingness)
• 15-question evaluation set with hand-checked answers and machine-checkable checks
• 12/15 plan accuracy, 93% tool, 85% parameter, 93% answer accuracy, 3/3 faithfulness (backup model)
• 281 offline tests, no API key needed
• Live app: https://data-detective-r.streamlit.app/
• Code: https://github.com/Rayyan134/data-detective

What I'm proudest of is the honesty machinery: an invented-number check that flags any statistic in the narrative that no tool computed; a PII redaction path that keeps identifier values out of every prompt; and an evaluation harness that measures five metrics on their own honest denominators, so a "not applicable" is never counted as a pass.

Honest limits (because they matter more than the wins):
• The authoritative primary-model run is still pending the free-tier daily quota (20 req/day/model).
• No charts were in the original plan — they're there now, but visualisations are basic.
• Latency is long-tailed (p95 ~15s); the UI shows a spinner only.
• Public deployment exists, but a full live smoke test is still on my list.

Huge thanks to the DDS Academy community and mentors for the structure and the feedback. Building in public for 8 days taught me more than any single course could.

#AI #LLM #AgenticAI #DataScience #Streamlit #BuildInPublic #DecodingDataScience

---

## Notes for the author

- The live app URL is https://data-detective-r.streamlit.app/ (verified HTTP 200).
- The repo is clean and all 281 tests pass.
- The demo video is skipped per your request; if you change your mind, a 60–90s screen recording of "load sample → ask a question → read findings vs answer → download report" is enough.
- The primary-model evaluation run is still pending the free-tier quota; the backup-model numbers (12/15 plan, 93% tool, 85% param, 93% answer, 3/3 faithful) are the ones to quote.
