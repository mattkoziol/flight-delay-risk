import datetime as dt
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Flight Delay Risk Predictor", page_icon="✈️", layout="wide")

BLUE = "#2a78d6"
FIRST, LAST, DEFAULT_DAY = dt.date(2026, 1, 1), dt.date(2026, 3, 31), dt.date(2026, 3, 13)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = {1: "January", 2: "February", 3: "March"}
BADGE = {"Low": ("green", ":material/check_circle:"), "Moderate": ("orange", ":material/warning:"),
         "High": ("red", ":material/error:")}
WHAT_IF = ["As recorded", "Inbound on time", "Inbound late by…", "Inbound cancelled or diverted"]
MODE = dict(zip(WHAT_IF, core.MODES))


@st.cache_resource(show_spinner="Loading 1.6 million flights and the model…")
def resources():
    flights = core.load_flights()
    dates = flights["FlightDate"].values
    starts = {d: (int(np.searchsorted(dates, np.datetime64(d), "left")), int(np.searchsorted(dates, np.datetime64(d), "right")))
              for d in pd.date_range(FIRST, LAST).date}
    route = flights.groupby(["Origin", "Dest"], observed=True)["delayed"].mean()
    day_net = flights.groupby("FlightDate")["p_hgb"].mean()
    return flights, starts, route, day_net, core.load_model()


@st.cache_data
def aggregates():
    return pd.read_parquet("data/agg.parquet")


@st.cache_data
def metrics():
    return json.load(open("data/metrics.json"))


flights, STARTS, ROUTE_RATE, DAY_NET, MODEL = resources()
AGG = aggregates()
HOUR_RATE = AGG.groupby("dep_hour")[["n", "n_delayed"]].sum().pipe(lambda d: d["n_delayed"] / d["n"])


def bar(df, x, y, title, sort=None, horizontal=False, fmt=".0%"):
    enc_cat = alt.Y(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=180)) if horizontal else alt.X(f"{x}:N", sort=sort, title=None,
                                                                              axis=alt.Axis(labelAngle=0))
    enc_val = (alt.X if horizontal else alt.Y)(f"{y}:Q", title=None, axis=alt.Axis(format=fmt, grid=True))
    tip = [alt.Tooltip(f"{x}:N", title=title.split(" by ")[-1].capitalize()),
           alt.Tooltip(f"{y}:Q", title="Delay rate", format=".1%"), alt.Tooltip("n:Q", title="Flights", format=",")]
    mark = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4)
    return mark.encode(**({"y": enc_cat, "x": enc_val} if horizontal else {"x": enc_cat, "y": enc_val}),
                       tooltip=tip).properties(title=title, height=300 if not horizontal else max(220, 26 * len(df)))


def rate(df, by):
    g = df.groupby(by, observed=True)[["n", "n_delayed"]].sum().reset_index()
    g["rate"] = g["n_delayed"] / g["n"]
    return g


def pct(p):
    return f"{p:.0%}"


def outcome_text(r):
    d = r["ArrDelay"]
    if np.isnan(d):
        return "No arrival time recorded."
    word = "early" if d < 0 else "late"
    verdict = "counted as **delayed** (15+ min)" if r["delayed"] else "counted as **on time**"
    return f"Arrived {abs(d):.0f} min {word}, {verdict}."


st.title("✈️ Flight Delay Risk Predictor")
st.caption("Real US domestic flights, January to March 2026 (BTS on-time data). Pick a flight to see the chance that it "
           "arrives 15 or more minutes late, scored live by the gradient-boosting model from Assignments 5 and 6.")

predict, explore, about = st.tabs(["Predict a flight", "Explore delays", "About the model"])

with predict:
    c1, c2, c3, c4 = st.columns([1, 1.3, 1.6, 1.6])
    day = c1.date_input("Date", DEFAULT_DAY, min_value=FIRST, max_value=LAST, key="date", format="MM/DD/YYYY")
    lo, hi = STARTS[day]
    today = flights.iloc[lo:hi]

    names = today.drop_duplicates("Reporting_Airline").set_index("Reporting_Airline")["carrier_name"].astype(str)
    airlines = today["Reporting_Airline"].value_counts().loc[lambda s: s > 0].index.astype(str).tolist()
    airline = c2.selectbox("Airline", airlines, key="airline", format_func=lambda a: f"{names[a]} ({a})")
    a_day = today[today["Reporting_Airline"] == airline]

    cities = a_day.drop_duplicates("Origin").set_index("Origin")["OriginCityName"].astype(str)
    origins = a_day["Origin"].value_counts().loc[lambda s: s > 0].index.astype(str).tolist()
    origin = c3.selectbox("From", origins, key="origin", format_func=lambda o: f"{o} · {cities[o]}")
    o_day = a_day[a_day["Origin"] == origin]

    dcities = o_day.drop_duplicates("Dest").set_index("Dest")["DestCityName"].astype(str)
    dests = o_day["Dest"].value_counts().loc[lambda s: s > 0].index.astype(str).tolist()
    dest = c4.selectbox("To", dests, key="dest", format_func=lambda d: f"{d} · {dcities[d]}")
    legs = o_day[o_day["Dest"] == dest].sort_values("CRSDepTime")

    if legs.empty:
        st.info("No scheduled flights match these choices. Try another date, airline or airport.")
        st.stop()

    base = [f"{airline} {r.Flight_Number_Reporting_Airline} · departs {core.fmt_hhmm(r.CRSDepTime)}, "
            f"arrives {core.fmt_hhmm(r.CRSArrTime)}" for r in legs.itertuples()]
    labels = {(t if base.count(t) == 1 else f"{t} ({n + 1})"): i for n, (t, i) in enumerate(zip(base, legs.index))}
    has_inbound = (legs["sched_turnaround"] > 0).values
    pick = st.selectbox(f"Departure ({len(legs)} on this route that day)", list(labels), key="flight",
                        index=int(has_inbound.argmax()) if has_inbound.any() else 0)
    row = legs.loc[[labels[pick]]]
    r = row.iloc[0]
    p = float(core.score(MODEL, row)[0])
    label = core.risk_label(p)
    route_rate = float(ROUTE_RATE.get((origin, dest), np.nan))

    st.divider()
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.metric("Chance of arriving 15+ min late", pct(p), f"{(p - route_rate) * 100:+.0f} pts vs this route's Q1 delay rate",
                  delta_color="inverse")
        color, icon = BADGE[label]
        st.badge(f"{label} risk", color=color, icon=icon)
        m1, m2, m3 = st.columns(3)
        m1.metric("Route, Q1", pct(route_rate), help="Actual delay rate on this route, January to March")
        m2.metric("Network", pct(float(DAY_NET[pd.Timestamp(day)])), help="Average model risk across all flights that day")
        m3.metric("This hour", pct(float(HOUR_RATE[r.dep_hour])),
                  help=f"Actual delay rate of all flights leaving in the {core.fmt_hhmm(r.dep_hour * 100)} hour, January to March")
    with right:
        st.markdown("**What the model saw**")
        st.markdown("\n".join(f"- {line}" for line in core.model_saw(r)))
        with st.expander("What actually happened"):
            st.markdown(outcome_text(r))
            st.caption("The model only used information available at the gate before departure, never this outcome.")

    st.subheader("What if the inbound plane were different?")
    no_inbound = not core.has_inbound(r)
    if no_inbound:
        choice = st.radio("Inbound aircraft", WHAT_IF, key="inbound_off", horizontal=True, disabled=True, index=0,
                          label_visibility="collapsed")
        st.caption("The model found no connecting inbound plane with time on the ground before this departure, "
                   "so there is nothing to change.")
    else:
        choice = st.radio("Inbound aircraft", WHAT_IF, key="inbound", horizontal=True, label_visibility="collapsed")
    minutes = 0
    if choice == WHAT_IF[2] and not no_inbound:
        minutes = st.slider("Minutes late", 0, 240, 60, step=5, key="late_min")
    if choice != WHAT_IF[0] and not no_inbound:
        p2 = float(core.score(MODEL, core.with_inbound(row, MODE[choice], minutes))[0])
        w1, w2 = st.columns([1, 2.4])
        w1.metric("What-if chance of 15+ min late", pct(p2), f"{(p2 - p) * 100:+.0f} pts vs recorded",
                  delta_color="inverse")
        w2.caption(f"Scheduled turnaround is {r['sched_turnaround']:.0f} min. If the inbound is later than that, the model "
                   "sees it as still overdue at departure. Re-scored live by the same model.")

    with st.expander(f"All {len(legs)} departures on {origin} → {dest} that day"):
        table = pd.DataFrame({
            "Flight": [f"{airline} {n}" for n in legs["Flight_Number_Reporting_Airline"]],
            "Departs": [core.fmt_hhmm(v) for v in legs["CRSDepTime"]],
            "Arrives": [core.fmt_hhmm(v) for v in legs["CRSArrTime"]],
            "Delay risk": core.score(MODEL, legs),
            "Actual arrival delay (min)": legs["ArrDelay"].values})
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Delay risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

with explore:
    f1, f2, f3 = st.columns(3)
    all_names = sorted(AGG["carrier_name"].unique())
    top_origins = AGG.groupby("Origin", observed=True)["n"].sum().sort_values(ascending=False).index.astype(str).tolist()
    sel_air = f1.multiselect("Airlines", all_names, key="ex_airline", placeholder="All airlines")
    sel_org = f2.multiselect("Departure airports", top_origins, key="ex_origin", placeholder="All airports")
    sel_mon = f3.multiselect("Months", list(MONTHS.values()), key="ex_month", placeholder="January to March")
    d = AGG
    if sel_air:
        d = d[d["carrier_name"].isin(sel_air)]
    if sel_org:
        d = d[d["Origin"].astype(str).isin(sel_org)]
    if sel_mon:
        d = d[d["month"].map(MONTHS).isin(sel_mon)]
    if d.empty:
        st.info("No flights match these filters. Clear one of them to see results.")
    else:

        by_hour = rate(d, "dep_hour")
        by_hour["hour"] = [core.fmt_hhmm(h * 100).replace(":00", "") for h in by_hour["dep_hour"]]
        by_hour = by_hour[by_hour["n"] >= 30]
        by_day = rate(d, "weekday").assign(day=lambda g: [WEEKDAYS[int(w) - 1] for w in g["weekday"]])
        by_air = rate(d, "carrier_name").sort_values("rate", ascending=False)
        by_org = rate(d, "Origin").sort_values("n", ascending=False).head(15).sort_values("rate", ascending=False)
        by_org["Origin"] = by_org["Origin"].astype(str)
        worst = by_hour.loc[by_hour["rate"].idxmax()] if len(by_hour) else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flights", f"{int(d['n'].sum()):,}")
        k2.metric("Arrived 15+ min late", pct(d["n_delayed"].sum() / d["n"].sum()))
        k3.metric("Worst departure hour", worst["hour"] if worst is not None else "–",
                  pct(worst["rate"]) + " delayed" if worst is not None else None, delta_color="off", delta_arrow="off")
        k4.metric("Best weekday", by_day.loc[by_day["rate"].idxmin(), "day"],
                  pct(by_day["rate"].min()) + " delayed", delta_color="off", delta_arrow="off")

        hour_order = [core.fmt_hhmm(h * 100).replace(":00", "") for h in range(24)]
        g1, g2 = st.columns(2)
        g1.altair_chart(bar(by_hour, "hour", "rate", "Delay rate by scheduled departure hour", sort=hour_order),
                        use_container_width=True)
        g2.altair_chart(bar(by_day, "day", "rate", "Delay rate by day of week", sort=WEEKDAYS), use_container_width=True)
        g3, g4 = st.columns(2)
        g3.altair_chart(bar(by_air, "carrier_name", "rate", "Delay rate by airline", sort="-x", horizontal=True),
                        use_container_width=True)
        g4.altair_chart(bar(by_org, "Origin", "rate", "Delay rate at the 15 busiest airports", sort="-x",
                            horizontal=True), use_container_width=True)
        st.caption("Delay = arrival 15 or more minutes after schedule. Cancelled and diverted flights are excluded. "
                   "Hours with fewer than 30 flights are hidden.")
        with st.expander("Table view"):
            st.dataframe(by_air.rename(columns={"carrier_name": "Airline", "n": "Flights", "n_delayed": "Delayed",
                                                "rate": "Delay rate"}), hide_index=True, use_container_width=True,
                         column_config={"Delay rate": st.column_config.NumberColumn(format="percent")})

with about:
    m = metrics()
    st.markdown("#### How good is the model?")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ROC AUC", f"{m['auc']:.3f}", help="Chance a random delayed flight scores above a random on-time one")
    a2.metric("PR AUC", f"{m['pr_auc']:.3f}", help=f"Baseline equals the test delay rate, {m['test_base_rate']:.3f}")
    a3.metric("Brier score", f"{m['brier']:.3f}", help="Mean squared error of the probabilities, lower is better")
    a4.metric("Calibration error", f"{m['ece']:.3f}", help="Expected calibration error: gap between predicted and "
              "observed delay rates, averaged over 15 bins")
    st.caption(f"Measured on the untouched test month, March 2026 ({m['rows']['test']:,} flights). For comparison, logistic "
               f"regression scored ROC AUC {m['lr_auc']:.3f} and random forest {m['rf_auc']:.3f} on the same test set (Assignment 6).")
    st.markdown(f"""
#### How it works
- **Model:** histogram gradient boosting (scikit-learn), trained in Assignment 6 on {m['rows']['train']:,} flights from
  {m['dates']['train'][0]} to {m['dates']['train'][1]}. Validation ran {m['dates']['valid'][0]} to {m['dates']['valid'][1]}
  and testing covered March. The split is by time, so the model is always scored on days after the ones it learned from.
- **Target:** whether the flight arrives 15 or more minutes late, the FAA/BTS definition of a delay.
- **What it uses (28 inputs):** airline, origin, destination, their states, day of week, departure hour, a holiday flag,
  scheduled departure and arrival times, block time and distance, how much padding the schedule has against the
  route's typical time, how busy both airports and the airline are that hour and day, the aircraft's leg number and
  scheduled turnaround, historical delay rates for the airline, airports, route and airport-hour (from the training
  months only), and the **inbound aircraft's status at the gate**, by far the strongest signal.
- **In this app:** picking a flight looks up its row in the 1.6 million-row schedule table and runs the model on it
  in a few milliseconds. The what-if control edits only the two inbound inputs and runs the model again.

#### Limits
- Only January 1 to March 31, 2026, because the model's inputs come from the real schedule. A future date has no
  schedule, congestion counts or inbound aircraft yet.
- Cancelled flights are not in the table, because the model predicts arrival delay for flights that operate.
- It scores the situation at the gate. Predictions made at booking time are weaker (ROC AUC about 0.68 in Assignment 5).
- Probabilities are calibrated on average (calibration error {m['ece']:.3f}) but any single flight can still go either way.
""")
