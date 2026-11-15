# Deployment (Day 7)

## Live deployment

**URL:** https://data-detective-r.streamlit.app/ (Streamlit Community Cloud, deployed 2026-11-15)

## Chosen target

**Streamlit Community Cloud** — free, deploys directly from the public GitHub
repository, and matches the stack (Streamlit + Python 3.12). No container
runtime is required for the demo, but a `Dockerfile` is provided for
self-hosting.

## Deploy steps (interactive — needs your Streamlit account)

1. Sign in at https://share.streamlit.io with the GitHub account that owns
   `Rayyan134/data-detective`.
2. Click **New app** → **From existing repo**.
3. Repository: `Rayyan134/data-detective`, branch `main`, main file `app.py`.
4. Python version: 3.12 (matches `requirements.txt`).
5. Under **Advanced settings → Secrets**, paste:
   ```
   GEMINI_API_KEY = "your-real-key"
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   ```
   (`.streamlit/secrets.toml` is git-ignored; Streamlit Cloud injects these as
   environment variables, which `src/llm_client.load_environment` reads.)
6. Deploy. The app URL will be
   `https://<app-name>-<user>.streamlit.app`.

## Self-host (Docker)

```powershell
docker build -t data-detective .
docker run -p 8501:8501 -e GEMINI_API_KEY=... -e GEMINI_MODEL=gemini-3.1-flash-lite data-detective
```

## Smoke test checklist (run on the deployed URL)

- [ ] App loads without an error banner.
- [ ] "Use the sample dataset" loads `sample_sales.csv` (180 rows, 11 columns).
- [ ] Uploading a CSV works; an invalid file shows a readable `DatasetError`.
- [ ] Asking a question shows the spinner, then findings + AI answer.
- [ ] An empty question disables the Investigate button.
- [ ] A missing API key surfaces as a readable error, not a crash.
- [ ] Redaction multiselect pre-selects identifier-like columns.
- [ ] The "Clear the result" button resets the report.
- [ ] Activity log is written (metadata only).

## Performance notes

- Free-tier quota: 20 requests/day/model; a full investigation costs 2.
- Latency: mean ~5.7s, p95 ~15.5s on the backup model (see `eval/evaluation.md` §5).
- The per-session investigation cache avoids repeat API calls for identical questions.
- Prompt tokens are ~92% of total usage; the obvious optimisation is trimming the tool catalogue prefix.

## Fallback plan if deployment breaks

- Keep the local run command working: `streamlit run app.py`.
- Screenshots of every major screen are committed under `docs/screenshots/`.
- The README documents the full setup, so a reviewer can run it locally in minutes.
