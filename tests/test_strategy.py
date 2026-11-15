"""Tests for strategy.py.  Run with:  pytest -q"""

import numpy as np
import pandas as pd
import pytest

from marketlens import strategy
from marketlens.data import load_data


def _synthetic_pair(n=1500, seed=0, cointegrated=True):
    """X is a random walk; Y = X^1.2 * exp(stationary noise) if cointegrated, else an independent walk."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    lx = np.log(100) + np.cumsum(rng.normal(0, 0.01, n))
    if cointegrated:
        e = np.zeros(n)
        for t in range(1, n):                     # AR(1) with phi = 0.9 -> half-life ~6.6 days
            e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.01)
        ly = 0.5 + 1.2 * lx + e
    else:
        ly = np.log(80) + np.cumsum(rng.normal(0, 0.01, n))
    return pd.Series(np.exp(ly), idx, name="Y"), pd.Series(np.exp(lx), idx, name="X")


@pytest.fixture(scope="module")
def prices():
    return load_data()[0]


def test_engle_granger_detects_cointegration():
    y, x = _synthetic_pair(cointegrated=True)
    eg = strategy.engle_granger(y, x)
    assert eg["p_value"] < 0.01
    assert eg["beta"] == pytest.approx(1.2, abs=0.05)
    assert eg["half_life_days"] == pytest.approx(-np.log(2) / np.log(0.9), rel=0.3)


def test_engle_granger_rejects_independent_random_walks():
    y, x = _synthetic_pair(cointegrated=False, seed=5)
    assert strategy.engle_granger(y, x)["p_value"] > 0.05


def test_positions_follow_rules():
    z = pd.Series([0, 2.5, 1.0, 0.3, -2.2, -1.0, 0.6, 1.5, 4.5, 3.0, 2.5, 0.2, 2.1])
    pos = strategy.generate_positions(z, entry=2.0, exit=0.5, stop=4.0).tolist()
    #       z:  0   2.5  1.0  0.3  -2.2 -1.0  0.6  1.5  4.5  3.0  2.5  0.2  2.1
    assert pos == [0, -1, -1, 0, 1, 1, 0, 0, 0, -1, -1, 0, -1]
    # 0.6 after a long: z crossed through zero -> exit.
    # 4.5 while flat: no entry beyond the stop level; 3.0 is back in the entry band -> short.


def test_stop_loss_then_lockout():
    z = pd.Series([2.5, 3.5, 4.2, 3.0, 2.5, 0.3, 2.4])
    pos = strategy.generate_positions(z, entry=2.0, exit=0.5, stop=4.0).tolist()
    # short at 2.5, stopped out at 4.2, locked out until |z|<0.5, then can re-enter
    assert pos == [-1, -1, 0, 0, 0, 0, -1]


def test_no_lookahead(prices):
    """Changing FUTURE prices must not change any PAST position or return."""
    y, x = prices["HDFCBANK"], prices["ICICIBANK"]
    base = strategy.backtest(y, x, "ols")
    cut = 900
    y2 = y.copy()
    y2.iloc[cut:] *= 1.5                            # huge shock from day `cut` onwards
    shocked = strategy.backtest(y2, x, "ols")
    pd.testing.assert_series_equal(base["position"].iloc[:cut + 1], shocked["position"].iloc[:cut + 1])
    pd.testing.assert_series_equal(base["net_ret"].iloc[:cut], shocked["net_ret"].iloc[:cut])


def test_execution_lag(prices):
    bt = strategy.backtest(prices["HDFCBANK"], prices["ICICIBANK"], "ratio", exec_lag=1)
    pd.testing.assert_series_equal(bt["position"].iloc[1:], bt["target"].shift(1).iloc[1:],
                                   check_names=False)


def test_costs_reduce_returns(prices):
    y, x = prices["HDFCBANK"], prices["ICICIBANK"]
    free = strategy.backtest(y, x, "ols", cost_bps=0)["equity"].iloc[-1]
    paid = strategy.backtest(y, x, "ols", cost_bps=20)["equity"].iloc[-1]
    assert paid < free


def test_gross_exposure_is_one_when_in_market(prices):
    bt = strategy.backtest(prices["HDFCBANK"], prices["ICICIBANK"], "ols")
    inmkt = bt[bt["position"] != 0]
    assert np.allclose(inmkt["w_y"].abs() + inmkt["w_x"].abs(), 1.0)


def test_trade_log_reconciles_with_equity(prices):
    bt = strategy.backtest(prices["HDFCBANK"], prices["ICICIBANK"], "ols")
    trades = strategy.trade_log(bt)
    assert (1 + trades["return"]).prod() == pytest.approx(bt["equity"].iloc[-1], rel=1e-10)


def test_strategy_profits_on_cointegrated_pair():
    """Sanity: on a truly mean-reverting synthetic pair, the rules should make money."""
    y, x = _synthetic_pair(cointegrated=True, seed=11)
    bt = strategy.backtest(y, x, "ols", cost_bps=0)
    assert bt["equity"].iloc[-1] > 1.0
