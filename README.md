# Flight Delay Risk Predictor

Pick a real US domestic flight from January to March 2026 and get the chance it arrives 15 or more minutes late,
scored live by a gradient-boosting model (ROC AUC 0.817 on the March 2026 test month). Includes an inbound-aircraft
what-if and a dashboard of delay patterns. IDC4290 Assignment 8.

**Live app:** https://flight-delay-risk-7f3c8ndjybutrhhgnuqwkg.streamlit.app/

## Run it locally
Python 3.12:

```bash
git clone https://github.com/mattkoziol/flight-delay-risk.git
cd flight-delay-risk
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Then open http://localhost:8501.

## Files
- `streamlit_app.py`: the web app (Predict, Explore, About tabs)
- `core.py`: lookup, inbound what-if, scoring, risk labels
- `data/flights.parquet`: 1,611,874 flights with the model's 28 inputs; `data/agg.parquet`: dashboard counts
- `models/hgb.joblib`: the trained model (scikit-learn 1.7.2)
- `prep.py`: rebuilds `data/` and `models/` from the Assignment 5 and 6 outputs (not needed to run the app)
- `tests/`: `python -m pytest` (38 tests)

Data: BTS Reporting Carrier On-Time Performance, January to March 2026.
