"""Pure logic for the Flight Delay Risk Predictor: lookup, inbound what-if, scoring, labels."""
CATS = ["carrier", "origin_top", "dest_top", "origin_state", "dest_state", "inb_status_gate"]
NUMS = ["weekday", "dep_hour", "holiday_near", "crs_dep_min", "crs_arr_min", "crs_elapsed", "distance", "route_pad",
        "orig_dep_hour_n", "orig_dep_day_n", "dest_arr_hour_n", "carrier_day_n", "leg_num", "legs_day",
        "sched_turnaround", "te_carrier", "te_origin", "te_dest", "te_route", "te_carrier_origin", "te_origin_hour",
        "inb_known_gate"]
FEATURES = CATS + NUMS

import joblib
import numpy as np
import pandas as pd

MODES = ("recorded", "on_time", "late", "disrupted")


def load_flights(path="data/flights.parquet"):
    return pd.read_parquet(path)


def load_model(path="models/hgb.joblib"):
    return joblib.load(path)


def score(model, rows):
    return model.predict_proba(rows[FEATURES])[:, 1]


def with_inbound(row, mode, minutes=0.0):
    """Copy of the rows with the at-gate inbound status replaced, following the assignment 5 rule at h = 0:
    an inbound later than the scheduled turnaround is still overdue at departure, and only the turnaround is known."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    out = row.copy()
    if mode == "recorded":
        return out
    turn = out["sched_turnaround"].astype(float)
    if turn.isna().any():
        raise ValueError("this flight has no inbound leg in the schedule")
    n = len(out)
    if mode == "on_time":
        status, known = np.full(n, "arrived"), np.zeros(n)
    elif mode == "late":
        late = (float(minutes) > turn).values
        status, known = np.where(late, "overdue", "arrived"), np.where(late, turn, float(minutes))
    else:
        status, known = np.full(n, "disrupted"), np.full(n, np.nan)
    out["inb_status_gate"] = pd.Series(status, index=out.index).astype(out["inb_status_gate"].dtype)
    out["inb_known_gate"] = pd.Series(known, index=out.index).astype(out["inb_known_gate"].dtype)
    return out


def risk_label(p):
    return "Low" if p < 0.15 else "Moderate" if p < 0.35 else "High"


def fmt_hhmm(v):
    v = int(v) % 2400
    h, m = divmod(v, 100)
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
