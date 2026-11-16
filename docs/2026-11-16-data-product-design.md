# Assignment 8 Data Product: Flight Delay Risk Predictor (design)

Due 2026-11-16 23:59. Graded on functionality, data/model integration, usability, scalability. Submission = a usable
tool (hosted link) + a user guide document.

## Intent
Turn the A5/A6 flight-delay work into a web app a grader can open from a link, look up a real Q1 2026 flight, get the
model's delay probability in real time, and explore delay patterns. Reuse, don't retrain.

## Approaches
- **A. Streamlit on Streamlit Community Cloud (chosen).** Fastest to build, deploys from a GitHub repo, best usability
  and dashboard. Cost: free tier sleeps (about 30 s cold start), about 1 GB memory, so RF (245 MB) is out and HGB is used.
- B. FastAPI on Render: good scalability story, poor usability for a grader (JSON forms).
- C. Gradio on Hugging Face Spaces: UI + API from one file, weaker dashboard, extra account.

## Goals
- Predict tab: date (Jan 1 to Mar 31 2026) -> airline -> origin -> destination -> pick a scheduled departure. Show HGB
  delay probability (P(arrival >= 15 min late)), a low/medium/high risk label, comparison with the route and day average,
  and what actually happened (it is historical data). What-if control: inbound aircraft lateness (on time / X minutes
  late / no inbound info), re-scored live by the model.
- Explore tab: filters (airline, origin, month) -> delay rate by hour, weekday, airline, top airports; KPI tiles.
- About the model tab: A6 test metrics for HGB (AUC 0.817, PR-AUC, calibration), features used, temporal split, limits.
- Scalability: model and data loaded once (`st.cache_resource` / `st.cache_data`), dashboard aggregates precomputed,
  single-row inference (milliseconds); guide explains how it would scale (batch scoring, API, refresh).

## Non-goals
- Predicting dates outside Q1 2026 or flights not in the BTS schedule (features depend on the real schedule).
- Retraining, new models, user accounts.

## Anti-goals
- A link that fails or times out for the grader. The app must fit Streamlit Cloud memory and start reliably.

## Decided
- Model: A6 `models/hgb.joblib` (sklearn 1.7.2, pinned in requirements). Inputs are the 28 A6 `FEATURES`; category dtypes
  must match training (`categorical_features="from_dtype"`).
- Data: a slim prep script builds `app/data/flights_app.parquet` from A5 `model.parquet` + `flights.parquet` (flight
  number, times, actual outcome) with float32 and categories, plus small aggregate parquets for the Explore tab.
  Every file under 100 MB (GitHub limit).
- Repo: new public GitHub repo under the user's logged-in account `mattkoziol`, folder `assignment8/`.
  Deploying on share.streamlit.io is a step the user does (sign in with GitHub, pick repo, click Deploy).
- User guide: Word doc in the A6/5A style (Times New Roman 12, AI-use statement first, numbered bold headings), with
  the link, step-by-step use, an example walkthrough, model notes, limits, and local-run fallback instructions.

## Left to the builder
Layout details, chart choices, risk-label cutoffs (calibrated from the score distribution), exact wording.
