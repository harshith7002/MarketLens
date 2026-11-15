"""Smoke test for the Streamlit dashboard: every tab must render without an exception.

Uses Streamlit's headless AppTest runner, so no browser or server is needed.
Slower than the other tests (~20 s) because the VaR backtest runs with 10,000 simulations.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=600).run()


def test_app_runs_without_exceptions(app):
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs] == ["Risk", "VaR", "Options", "Pairs Trading", "Rates & Credit"]


def test_app_numbers_match_library(app):
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["99% VaR, historical"] == "Rs 247,071"
    assert metrics["Price"] == "10.4506"          # Black-Scholes ATM call
    assert metrics["Total return"] == "9.82%"     # pairs, out-of-sample
    assert metrics["Bond price"] == "100.0000"    # 10y 7% bond at a 7% yield
    assert metrics["Modified duration"] == "7.106"
    assert metrics["Parallel DV01"] == "$5,293 / bp"          # Book A on 2026-11-12
    assert metrics["CDS par spread"] == "111.68 bp"           # US BBB, 5y


def test_changing_inputs_still_works(app):
    put = [r for r in app.radio if r.options == ["call", "put"]][0]
    put.set_value("put").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Price"] == "5.5735"
