# Day 5 LinkedIn post — Interface integration

Screenshot to attach: `docs/screenshots/day5-interface.png`

---

**Draft post**

Day 5 of the DDS AI Application Building Challenge: the brain is now in the browser. 🖥️

"Data Detective" used to be two halves — a profiling UI and an agent that only ran from the command line. Today they became one app.

Upload a CSV (or load the sample), ask a question in plain English, and get back:
📊 computed findings (real tables, real numbers)
🤖 an AI answer that is clearly labelled as interpretation, not evidence
⚠️ a warning if the narrative contains a number no tool actually computed

The UX details that matter:
• Example questions are generated from the *actual* columns of your dataset — no hardcoded demo questions.
• Columns that look like identifiers (order_id, customer, email…) are pre-selected for PII redaction, so their values never reach the model. You can change the selection.
• A loading spinner covers the ~10s a question takes, plus friendly error states (missing key, rate limit, malformed plan) and a reset/clear button.
• Every run is logged as one JSON line — metadata only: question, shape, tools, params, status, tokens, latency. No cell values, no API key.

The thing I'm most proud of: the interface layer is split from the rendering. `src/ui_report.py` turns a report into plain data and `src/audit.py` handles logging, both with no Streamlit import, so they're covered by 39 new offline tests. The app itself is exercised end-to-end with Streamlit's AppTest (no browser needed). 271 tests pass offline.

Honest note: the screenshot is of a real run on the backup model; the public demo link arrives on Day 7 (deployment). Locally it's one command: `streamlit run app.py`.

Code: https://github.com/Rayyan134/data-detective

#AI #LLM #AgenticAI #Streamlit #DataScience #BuildInPublic #DecodingDataScience

---

## Notes for the author

- Screenshot is the Day 5 interface mock (`docs/screenshots/day5-interface.html` → PNG).
- Numbers: 271 tests, 39 new (ui_report, audit, app).
- The public demo link is deferred to Day 7 (deployment); the local run command is the Day 5 "rough demo".
