# Day 7 LinkedIn post — Visualisations & deployment

Screenshot to attach: `docs/screenshots/day7-visualizations.png`

---

**Draft post**

Day 7 of the DDS AI Application Building Challenge: the last planned feature, plus deployment prep. 📊

"Data Detective" now draws charts — but only from computed results. Bars, lines and heatmaps come from the tool result tables (altair), never from model text. The model still only plans and writes the narrative; Python still does every number and every pixel.

What shipped today:
• `src/visualizer.py` — chart builders for 5 of the 6 tools (group comparison, value counts, trend, missingness, correlation heatmap). `find_outliers` stays table-only.
• Charts render under each finding in the Streamlit report, via `st.altair_chart`.
• Deployment prep: a Streamlit theme config, a `Dockerfile`, and `docs/deployment.md` with step-by-step Streamlit Cloud instructions, a smoke-test checklist, performance notes and a fallback plan.

Honest notes:
• The app is live: https://data-detective-r.streamlit.app/ (Streamlit Community Cloud). It also runs locally with `streamlit run app.py`, and every screen is captured under `docs/screenshots/`.
• Free-tier quota (20 req/day/model) still blocks the authoritative primary-model evaluation run.
• 281 tests pass offline.

Code: https://github.com/Rayyan134/data-detective

#AI #LLM #AgenticAI #DataVisualisation #Streamlit #DataScience #BuildInPublic #DecodingDataScience

---

## Notes for the author

- Screenshot is the Day 7 visualisations/deployment card (`docs/screenshots/day7-visualizations.html` → PNG).
- The app is live at https://data-detective-r.streamlit.app/ (deployed 2026-11-15).
- Day 8 will be the final submission + LinkedIn sharing.
