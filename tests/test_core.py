import os

import numpy as np
import pandas as pd
import pytest

import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def flights():
    return core.load_flights(os.path.join(ROOT, "data", "flights.parquet"))


@pytest.fixture(scope="module")
def model():
    return core.load_model(os.path.join(ROOT, "models", "hgb.joblib"))


def one_row(flights, turnaround):
    r = flights[flights["sched_turnaround"].notna()].head(1).copy()
    r["sched_turnaround"] = np.float32(turnaround)
    return r


@pytest.mark.parametrize("mode,minutes,status,known", [
    ("late", 30, "arrived", 30), ("late", 90, "overdue", 45), ("late", 45, "arrived", 45),
    ("on_time", 0, "arrived", 0), ("disrupted", 0, "disrupted", np.nan)])
def test_with_inbound(flights, mode, minutes, status, known):
    r = core.with_inbound(one_row(flights, 45), mode, minutes)
    assert r["inb_status_gate"].iloc[0] == status
    assert (np.isnan(known) and np.isnan(r["inb_known_gate"].iloc[0])) or r["inb_known_gate"].iloc[0] == known
    assert r["inb_status_gate"].dtype == flights["inb_status_gate"].dtype


def test_with_inbound_recorded_is_copy(flights):
    r = one_row(flights, 45)
    out = core.with_inbound(r, "recorded")
    assert out.equals(r) and out is not r


def test_with_inbound_without_inbound_raises(flights):
    r = flights[flights["sched_turnaround"].isna()].head(1)
    with pytest.raises(ValueError):
        core.with_inbound(r, "late", 30)
    assert core.with_inbound(r, "recorded").equals(r)


def test_with_inbound_does_not_mutate(flights):
    r = one_row(flights, 45)
    before = r.copy()
    core.with_inbound(r, "late", 120)
    assert r.equals(before)


def test_score_matches_stored(flights, model):
    s = flights.sample(500, random_state=1)
    assert np.abs(core.score(model, s) - s["p_hgb"].values).max() < 1e-6


def test_late_inbound_raises_risk(flights, model):
    r = flights[(flights["sched_turnaround"] >= 60)].sample(1, random_state=2)
    assert core.score(model, core.with_inbound(r, "late", 120))[0] > core.score(model, core.with_inbound(r, "on_time"))[0]


@pytest.mark.parametrize("p,label", [(0.0, "Low"), (0.149, "Low"), (0.15, "Moderate"), (0.349, "Moderate"),
                                     (0.35, "High"), (1.0, "High")])
def test_risk_label(p, label):
    assert core.risk_label(p) == label


@pytest.mark.parametrize("v,s", [(705, "7:05 AM"), (0, "12:00 AM"), (2400, "12:00 AM"), (1330, "1:30 PM"),
                                 (1200, "12:00 PM"), (2359, "11:59 PM"), (59, "12:59 AM")])
def test_fmt_hhmm(v, s):
    assert core.fmt_hhmm(v) == s


def test_zero_turnaround_has_no_what_if(flights):
    r = one_row(flights, 0)
    assert not core.has_inbound(r.iloc[0])
    with pytest.raises(ValueError):
        core.with_inbound(r, "late", 30)
    assert core.has_inbound(one_row(flights, 45).iloc[0])


def test_inbound_text_without_connecting_leg(flights):
    r = flights[flights["sched_turnaround"].isna() & (flights["leg_num"] > 1)].iloc[0]
    text = core.inbound_text(r)
    assert "first flight" not in text and "no connecting inbound" in text.lower()


def test_model_saw_has_no_nan(flights):
    r = flights[flights["crs_elapsed"].isna()].iloc[0]
    assert "nan" not in " ".join(core.model_saw(r)).lower()


def test_load_peak_memory_under_750mb():
    import subprocess, sys
    code = ("import resource, sys, core; core.load_flights(); core.load_model();"
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == 'darwin' else 1e3))")
    mb = float(subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    assert mb < 750, mb
