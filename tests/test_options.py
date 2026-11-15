"""Tests for options.py.  Run with:  pytest -q"""

import numpy as np
import pytest

from marketlens import options as o

EX = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def test_bs_matches_textbook_value():
    # Hull's classic example values: S=K=100, T=1, r=5%, sigma=20%
    assert float(o.bs_price(**EX, kind="call")) == pytest.approx(10.4506, abs=1e-4)
    assert float(o.bs_price(**EX, kind="put")) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("K", [70, 90, 100, 115, 140])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(K, T, q):
    """C - P = S e^{-qT} - K e^{-rT} must hold for any strike/maturity."""
    S, r, sig = 100.0, 0.05, 0.25
    c = o.bs_price(S, K, T, r, sig, "call", q)
    p = o.bs_price(S, K, T, r, sig, "put", q)
    assert float(c - p) == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_monte_carlo_within_1pct_at_100k_paths():
    bs = float(o.bs_price(**EX, kind="call"))
    mc, se = o.mc_price(**EX, kind="call", n_paths=100_000, seed=42)
    assert abs(mc - bs) / bs < 0.01
    assert abs(mc - bs) < 3 * se  # and consistent with its own standard error


def test_antithetic_reduces_standard_error():
    _, se_plain = o.mc_price(**EX, n_paths=100_000, seed=1)
    _, se_anti = o.mc_price(**EX, n_paths=100_000, seed=1, antithetic=True)
    assert se_anti < se_plain


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = o.bs_greeks(**EX, kind=kind)
    S, K, T, r, s = EX.values()
    h = 1e-4
    price = lambda **kw: float(o.bs_price(**{**EX, **kw}, kind=kind))
    assert g["delta"] == pytest.approx((price(S=S + h) - price(S=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((price(S=S + h) - 2 * price() + price(S=S - h)) / h**2, rel=1e-3)
    assert g["vega"] * 100 == pytest.approx((price(sigma=s + h) - price(sigma=s - h)) / (2 * h), rel=1e-5)
    assert g["rho"] * 100 == pytest.approx((price(r=r + h) - price(r=r - h)) / (2 * h), rel=1e-5)
    # theta = dV/dt = -dV/dT, per calendar day
    assert g["theta"] * 365 == pytest.approx(-(price(T=T + h) - price(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("K,T,sig", [(100, 1.0, 0.2), (80, 0.25, 0.45), (130, 2.0, 0.15), (95, 0.02, 0.6)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(K, T, sig, kind):
    price = float(o.bs_price(100, K, T, 0.03, sig, kind))
    res = o.implied_vol(price, 100, K, T, 0.03, kind)
    assert res.sigma == pytest.approx(sig, abs=1e-8)


def test_implied_vol_bisection_fallback_works():
    price = float(o.bs_price(100, 120, 0.5, 0.03, 0.35, "call"))
    res = o.implied_vol(price, 100, 120, 0.5, 0.03, "call", max_iter=1)  # starve Newton
    assert res.method == "bisection"
    assert res.sigma == pytest.approx(0.35, abs=1e-7)


def test_implied_vol_rejects_arbitrage_and_no_time_value():
    assert o.implied_vol(150.0, 100, 100, 1, 0.05).method == "no-arbitrage violation"  # call > stock
    assert np.isnan(o.implied_vol(1e-20, 100, 200, 0.05, 0.05).sigma)


def test_delta_hedge_pnl_small_and_unbiased():
    """Daily hedging with the right vol: P&L ~ 0 on average, small vs premium,
    and independent of the real-world drift (mu)."""
    rng = np.random.default_rng(0)
    T = 60 / 252
    for mu in (0.0, 0.30):
        paths = o.gbm_paths(100, T, 0.05, 0.2, 60, 10_000, rng, mu=mu)
        h = o.delta_hedge(paths, 100, T, 0.05, 0.2)
        assert abs(h["pnl"].mean()) < 0.03            # premium is ~4.5
        assert h["pnl"].std() < 0.12 * h["premium"].mean()


def test_more_frequent_hedging_reduces_error():
    rng = np.random.default_rng(3)
    T = 60 / 252
    paths = o.gbm_paths(100, T, 0.05, 0.2, 60, 10_000, rng)
    daily = o.delta_hedge(paths, 100, T, 0.05, 0.2)["pnl"].std()
    weekly = o.delta_hedge(paths[::5], 100, T, 0.05, 0.2)["pnl"].std()
    assert daily < weekly
    # error ~ 1/sqrt(rebalances): 5x more rebalances -> ~sqrt(5) = 2.24x smaller
    assert weekly / daily == pytest.approx(np.sqrt(5), rel=0.15)
