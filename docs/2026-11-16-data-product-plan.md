# Flight Delay Risk Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app on Streamlit Community Cloud that scores real Q1 2026 flights with the A6 HGB model, supports an inbound-aircraft what-if, and has an Explore dashboard and an About-the-model tab; plus a Word user guide.

**Architecture:** `prep.py` (run locally once) slims the A5/A6 artifacts into `data/` and `models/` inside `assignment8/`, which is its own git repo pushed to GitHub. `core.py` holds pure, tested logic (lookup, what-if feature edits, scoring, risk labels). `streamlit_app.py` is UI only and caches data and model.

**Tech Stack:** Python 3.12, streamlit 1.67.0, altair (ships with streamlit), scikit-learn 1.7.2, pandas 2.2.3, numpy 2.2.6, pyarrow, joblib 1.5.2, pytest; python-docx for the guide.

**Spec:** `docs/2026-11-16-data-product-design.md`

## Global Constraints
- Due 2026-11-16 23:59. Keep scope to the spec.
- Model: A6 `models/hgb.joblib`, 28 `FEATURES` from `assignment6/common.py` (6 CATS + 22 NUMS), category dtypes exactly as in `assignment5/data/clean/model.parquet`.
- Every committed file < 95 MB; app resident memory target < 700 MB.
- requirements.txt pins `scikit-learn==1.7.2`, `pandas==2.2.3`, `numpy==2.2.6`, `joblib==1.5.2`, `streamlit==1.67.0`, `pyarrow`.
- Target is P(arrival >= 15 min late), the A5 `delayed` column. Dates Jan 1 to Mar 31 2026.
- Risk labels: Low < 0.15, Moderate 0.15 to < 0.35, High >= 0.35 (base rate 21.8%).
- No code comments except non-obvious steps (user preference).

## Review Focus
- Route with one departure that day, or a date/airline/origin combo with no flights: selectors must narrow without crashing (empty selectbox handled with a message).
- Flight with no inbound leg (`sched_turnaround` NaN): what-if control disabled with a short explanation, score shown as recorded.
- Inbound "X minutes late" larger than the scheduled turnaround: must become `overdue` with `inb_known_gate = sched_turnaround` (A5 `inbound_at_horizon` rule at h=0), not `arrived`.
- Cold start on the cloud: data loaded once via `st.cache_resource`; first render under the platform timeout.
- Cancelled flights are not in the model table (A5 excluded them); guide must say so, and the lookup must not offer them.

---

### Task 1: Data and model export (`prep.py`)

**Files:**
- Create: `assignment8/prep.py`, `assignment8/tests/test_prep.py`
- Produces files: `data/flights.parquet`, `data/agg.parquet`, `data/metrics.json`, `models/hgb.joblib`

**Interfaces:**
- Produces `data/flights.parquet`, one row per `model.parquet` row, columns: `row_id`, `FlightDate`, `split`, `delayed`, the 28 FEATURES, and display columns from A5 `flights.parquet` joined on `row_id`: `Reporting_Airline`, `carrier_name`, `Flight_Number_Reporting_Airline`, `Origin`, `OriginCityName`, `Dest`, `DestCityName`, `CRSDepTime`, `CRSArrTime`, `ArrDelay`, plus `p_hgb` from `assignment6/data/preds.parquet`. Numeric features float32 except `te_*`, `route_pad` float64 (keep whichever set makes the parity test pass); string display columns as category.
- Produces `data/agg.parquet`: rows grouped by `month, Reporting_Airline, Origin, weekday, dep_hour` with `n` and `n_delayed` (Explore tab re-aggregates from this).
- Produces `data/metrics.json`: HGB test-split `auc`, `pr_auc`, `brier`, `ece`, base rate, and row counts per split, read from A6 `output/` files where present, else computed from `p_hgb` vs `delayed` on `split == "test"`.

- [ ] Step 1: `test_parity`: load `data/flights.parquet` and `models/hgb.joblib`, score a random 20,000-row sample (seed 0); `max|p - p_hgb| < 1e-6`.
- [ ] Step 2: `test_sizes`: every file under `data/` and `models/` < 95 MB; `test_columns`: categories of each CATS column equal those in A5 `model.parquet`.
- [ ] Step 3: Implement `prep.py`; run `python3 prep.py`; run `python3 -m pytest tests/test_prep.py -q` -> pass.
- [ ] Step 4: `git init`, `.gitignore` (`__pycache__`, `.DS_Store`), commit.

### Task 2: Core logic (`core.py`)

**Files:** Create `assignment8/core.py`, `assignment8/tests/test_core.py`

**Interfaces:**
- Consumes Task 1 files.
- `FEATURES: list[str]`, `CATS: list[str]` (copied from A6 common.py, not imported).
- `load_flights(path="data/flights.parquet") -> pd.DataFrame`; `load_model(path="models/hgb.joblib")`.
- `score(model, rows: pd.DataFrame) -> np.ndarray` (P(delayed) per row, uses `rows[FEATURES]`).
- `with_inbound(row: pd.DataFrame, mode: str, minutes: float = 0) -> pd.DataFrame` — one-row copy with `inb_status_gate`, `inb_known_gate` edited; `mode` in `"recorded" | "on_time" | "late" | "disrupted"`. `on_time` -> arrived, 0. `late` -> arrived with `minutes` if `minutes <= sched_turnaround`, else overdue with `sched_turnaround`. `disrupted` -> disrupted, NaN. Raises `ValueError` if mode != recorded and `sched_turnaround` is NaN. Categorical dtype preserved.
- `risk_label(p: float) -> str` -> `"Low" | "Moderate" | "High"` per Global Constraints.
- `fmt_hhmm(v) -> str` -> `"7:05 AM"` style from HHMM numbers (A5/A6 report style uses AM/PM).

- [ ] Step 1: Tests: `with_inbound` on a row with turnaround 45: late 30 -> ("arrived", 30), late 90 -> ("overdue", 45), on_time -> ("arrived", 0), disrupted -> ("disrupted", NaN); NaN turnaround raises; dtype of `inb_status_gate` unchanged. `risk_label(0.149)=="Low"`, `(0.15)=="Moderate"`, `(0.35)=="High"`. `fmt_hhmm(705)=="7:05 AM"`, `(0)` and `(2400)=="12:00 AM"`, `(1330)=="1:30 PM"`. On real data: late 120 on a sampled flight with an inbound scores higher than on_time.
- [ ] Step 2: Run -> fail. Implement. Run `python3 -m pytest tests -q` -> pass. Commit.

### Task 3: Streamlit app (`streamlit_app.py`)

**Files:** Create `assignment8/streamlit_app.py`, `assignment8/requirements.txt`, `assignment8/.streamlit/config.toml`, `assignment8/tests/test_app.py`

**Interfaces:** Consumes everything in Task 2. Load the `dataviz` skill before writing charts.

- Sidebar-free layout, title "Flight Delay Risk Predictor", three `st.tabs`: Predict, Explore, About the model.
- Predict: date input (min 2026-01-01, max 2026-03-31, default 2026-03-13), airline, origin, destination selectboxes each filtered by the previous choice, then a departure picker labeled `"<carrier><flight no>  <dep> -> <arr>"`. Result: big probability metric with risk label, delta vs route average and vs that day's network average, a short "why" line from the inbound status and congestion counts, "What actually happened" (arrival delay minutes, delayed yes/no). What-if radio (recorded / on time / late with slider 0 to 240 min / cancelled or diverted) re-scores live via `with_inbound` + `score`.
- Explore: multiselect airline, origin, month; KPI row (flights, delay rate, worst hour); altair charts of delay rate by departure hour, by weekday, by airline, top 15 origins; from `agg.parquet`.
- About: metrics from `metrics.json`, feature list in plain words, temporal split, limits (Q1 2026 only, cancelled excluded, historical lookup not live feeds).
- `@st.cache_resource` for model and flights table, `@st.cache_data` for agg.

- [ ] Step 1: `test_app.py` with `streamlit.testing.v1.AppTest.from_file("streamlit_app.py", default_timeout=120)`: runs with no exception; Predict tab shows a metric containing "%"; switching the what-if radio to "late" and slider to 120 changes the metric value.
- [ ] Step 2: `pip3 install streamlit==1.67.0`; implement; run tests -> pass. `streamlit run streamlit_app.py` locally and check in the browser (screenshot each tab). Check RSS memory of the server process < 700 MB.
- [ ] Step 3: Commit.

### Task 4: Publish

- [ ] Step 1: `README.md` (what it is, link placeholder, local run: `pip install -r requirements.txt && streamlit run streamlit_app.py`).
- [ ] Step 2: `gh repo create mattkoziol/flight-delay-risk --public --source . --push`.
- [ ] Step 3: User deploys on share.streamlit.io (repo `mattkoziol/flight-delay-risk`, branch `main`, file `streamlit_app.py`, Python 3.12 in advanced settings) and pastes the URL; verify it loads and predicts.

### Task 5: User guide (Word)

**Files:** Create `assignment8/guide/Assignment8_User_Guide.md`, `assignment8/guide/build_docx.py` -> `Assignment8_User_Guide_Koziol.docx`

- Styled like A6 (`assignment6/10_build_docx.py`: Times New Roman 12, AI-use statement first per memory feedback, bold numbered headings, "- Label:" bullets). Sections: 1 What the tool does, 2 Opening it (link, cold start note, local fallback), 3 Predicting a flight (steps + worked example with real numbers and screenshots from Task 3), 4 What-if inbound, 5 Explore tab, 6 About the model and how the data/model are integrated, 7 Scalability (caching, precomputed aggregates, ms inference, how it would grow: API, batch scoring, live feeds), 8 Limits.
- [ ] Step 1: Write md, build docx, open and check. Commit and push.
