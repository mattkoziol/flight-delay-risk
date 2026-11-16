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
import pyarrow.parquet as pq

MODES = ("recorded", "on_time", "late", "disrupted")


def load_flights(path="data/flights.parquet"):
    """Arrow-to-pandas conversion that frees each Arrow column as it goes, keeping peak memory near the final size."""
    cols = [c for c in pq.read_schema(path).names if c not in ("row_id", "split")]
    return pq.read_table(path, columns=cols).to_pandas(self_destruct=True, split_blocks=True)


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
    if not (turn > 0).all():
        raise ValueError("this flight has no inbound leg with a positive turnaround in the schedule")
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


def has_inbound(r):
    """True when the what-if applies: a connecting inbound leg with time on the ground before departure."""
    t = r["sched_turnaround"]
    return bool(pd.notna(t) and t > 0)


def inbound_text(r):
    s, k, t = r["inb_status_gate"], r["inb_known_gate"], r["sched_turnaround"]
    if s == "none":
        if r["leg_num"] == 1:
            return "This is the aircraft's first flight of the day, so there is no inbound plane."
        return ("No connecting inbound leg found: the aircraft's previous flight that day ended at another airport "
                "or its tail number is missing.")
    turn = f"{t:.0f}-minute scheduled turnaround"
    if s == "arrived":
        when = "exactly on time" if k == 0 else f"{abs(k):.0f} min {'early' if k < 0 else 'late'}"
        return f"Inbound aircraft arrived {when}, with a {turn}."
    if s == "overdue":
        return f"Inbound aircraft still had not arrived at departure time (more than {k:.0f} min late), {turn}."
    if s == "disrupted":
        return f"Inbound aircraft's previous leg was cancelled or diverted ({turn})."
    return f"Inbound aircraft was not yet due at departure time ({turn})."


def model_saw(r):
    lines = [inbound_text(r), f"{r['orig_dep_hour_n']:.0f} departures scheduled at {r['Origin']} in the same hour "
                              f"({r['orig_dep_day_n']:.0f} that day)."]
    leg = f"Leg {r['leg_num']:.0f} of {r['legs_day']:.0f} for this aircraft today." if pd.notna(r["leg_num"]) else ""
    block = (f"Scheduled block time {r['crs_elapsed']:.0f} min for {r['distance']:.0f} miles."
             if pd.notna(r["crs_elapsed"]) else f"Distance {r['distance']:.0f} miles.")
    lines.append(f"{leg} {block}".strip())
    return lines
