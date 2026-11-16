import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def at(monkeypatch):
    monkeypatch.chdir(ROOT)
    a = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=120)
    a.run()
    assert not a.exception, a.exception
    return a


def risk(at):
    return next(m.value for m in at.metric if m.label.startswith("Chance"))


def test_predict_shows_probability(at):
    assert risk(at).endswith("%")
    assert len(at.tabs) == 3


def test_what_if_late_changes_risk(at):
    base = risk(at)
    at.radio(key="inbound").set_value("Inbound late by…").run()
    at.slider(key="late_min").set_value(150).run()
    assert not at.exception
    assert next(m.value for m in at.metric if m.label.startswith("What-if")) != base


def test_changing_route_keeps_working(at):
    sb = at.selectbox(key="airline")
    sb.select_index(len(sb.options) - 1).run()
    assert not at.exception
    assert risk(at).endswith("%")


def test_explore_filters(at):
    ms = at.multiselect(key="ex_airline")
    ms.set_value([ms.options[0]]).run()
    assert not at.exception
    assert any(m.label == "Flights" for m in at.metric)
