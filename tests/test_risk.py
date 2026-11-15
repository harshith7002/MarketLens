"""Tests for data.py and risk.py. Run with:  pytest -q"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketlens import risk
from marketlens.data import load_data, log_returns


@pytest.fixture(scope="module")
def data():
    return load_data()  # cached CSVs - runs offline


@pytest.fixture(scope="module")
def synthetic():
    """Jointly normal returns with known parameters, for checking formulas."""
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]]) * 0.01**2
    x = rng.multivariate_normal([0, 0], cov, size=200_000)
    return pd.DataFrame(x, columns=["A", "B"])


def test_no_nans_in_returns(data):
    _, rets = data
    assert not rets.isna().any().any()
    assert np.isfinite(rets.to_numpy()).all()


def test_log_returns_definition():
    p = pd.DataFrame({"X": [100.0, 110.0, 99.0]})
    r = log_returns(p)["X"].to_numpy()
    assert r == pytest.approx([np.log(1.1), np.log(0.9)])


@pytest.mark.parametrize("method", ["Historical", "Parametric", "Monte Carlo"])
def test_var_ordering_95_below_99(data, method):
    _, rets = data
    vt = risk.var_table(rets).set_index(["Method", "Level"])
    assert vt.loc[(method, "95%"), "VaR %"] < vt.loc[(method, "99%"), "VaR %"]
    # Expected Shortfall is the average beyond VaR, so it can't be smaller
    for lvl in ("95%", "99%"):
        assert vt.loc[(method, lvl), "ES %"] >= vt.loc[(method, lvl), "VaR %"]


def test_parametric_var_matches_closed_form(synthetic):
    # Portfolio 50/50 of two assets with sd 1%, corr 0.5 -> sd_p = 1% * sqrt(0.75)
    w = {"A": 0.5, "B": 0.5}
    var, _ = risk.parametric_var_es(np.log1p(synthetic), w, 0.99)
    expected = stats.norm.ppf(0.99) * 0.01 * np.sqrt(0.75)
    assert var == pytest.approx(expected, rel=0.02)


def test_monte_carlo_close_to_parametric(synthetic):
    w = {"A": 0.5, "B": 0.5}
    lr = np.log1p(synthetic)
    p_var, _ = risk.parametric_var_es(lr, w, 0.95)
    mc_var, _ = risk.monte_carlo_var_es(lr, w, 0.95, n_sims=100_000)
    assert mc_var == pytest.approx(p_var, rel=0.03)


def test_kupiec_perfect_calibration():
    lr, p = risk.kupiec_pof(breaches=5, n_obs=500, level=0.99)  # exactly 1%
    assert lr == pytest.approx(0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_far_too_many_breaches():
    _, p = risk.kupiec_pof(breaches=15, n_obs=250, level=0.99)  # 6% vs 1%
    assert p < 0.01


def test_kupiec_zero_breaches_is_finite():
    lr, p = risk.kupiec_pof(breaches=0, n_obs=250, level=0.99)
    assert np.isfinite(lr) and 0 <= p <= 1


def test_max_drawdown_simple():
    p = pd.Series([100, 120, 90, 130, 65])
    assert risk.max_drawdown(p) == pytest.approx(65 / 130 - 1)


def test_backtest_has_no_lookahead(data):
    """Changing the realised return on day t must not change the VaR forecast for day t."""
    _, rets = data
    rets = rets.iloc[-300:]
    base = risk.rolling_var_forecasts(rets, test_days=5, window=250, n_sims=1000)
    shocked = rets.copy()
    shocked.iloc[-1] = -0.5  # crash on the last day
    after = risk.rolling_var_forecasts(shocked, test_days=5, window=250, n_sims=1000)
    pd.testing.assert_series_equal(base["Historical_99"], after["Historical_99"])
    pd.testing.assert_series_equal(base["Parametric_99"], after["Parametric_99"])
