import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from core import CATS, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A5 = os.path.join(ROOT, "..", "assignment5", "data", "clean", "model.parquet")


@pytest.fixture(scope="module")
def flights():
    return pd.read_parquet(os.path.join(ROOT, "data", "flights.parquet"))


def test_parity(flights):
    model = joblib.load(os.path.join(ROOT, "models", "hgb.joblib"))
    s = flights.sample(20_000, random_state=0)
    p = model.predict_proba(s[FEATURES])[:, 1]
    assert np.abs(p - s["p_hgb"].values).max() < 1e-6


def test_sizes():
    for d in ("data", "models"):
        for f in os.listdir(os.path.join(ROOT, d)):
            assert os.path.getsize(os.path.join(ROOT, d, f)) < 95e6, f


@pytest.mark.skipif(not os.path.exists(A5), reason="assignment 5 table not present")
def test_columns(flights):
    src = pd.read_parquet(A5, columns=CATS)
    for c in CATS:
        assert list(flights[c].cat.categories) == list(src[c].cat.categories), c


def test_metrics_and_agg():
    m = json.load(open(os.path.join(ROOT, "data", "metrics.json")))
    assert 0.81 < m["auc"] < 0.83 and 0.2 < m["base_rate"] < 0.23
    agg = pd.read_parquet(os.path.join(ROOT, "data", "agg.parquet"))
    assert {"month", "Reporting_Airline", "Origin", "weekday", "dep_hour", "n", "n_delayed"} <= set(agg.columns)
    assert agg["n"].sum() == 1_611_874
