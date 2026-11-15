"""Tests for fixed_income.py.  Run with:  pytest -q"""

import numpy as np
import pytest

from marketlens import fixed_income as fi


# ---------------------------------------------------------------------------
# Part 1 - bond pricing and risk measures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 5, 10, 30])
@pytest.mark.parametrize("y", [0.01, 0.07, 0.15])
def test_zero_coupon_duration_equals_maturity(m, y):
    assert fi.macaulay_duration(y, 0.0, m) == pytest.approx(m, abs=1e-12)


def test_zero_coupon_price_closed_form():
    assert fi.bond_price(0.07, 0.0, 10) == pytest.approx(100 / 1.035**20, rel=1e-14)


@pytest.mark.parametrize("c,m", [(0.07, 2), (0.07, 10), (0.0, 10), (0.04, 30)])
def test_price_falls_when_yield_rises(c, m):
    ys = np.linspace(0.0, 0.20, 81)
    prices = [fi.bond_price(v, c, m) for v in ys]
    assert np.all(np.diff(prices) < 0)


@pytest.mark.parametrize("c", [0.03, 0.07, 0.12])
def test_par_bond_prices_at_100(c):
    """When the coupon equals the yield, the bond is worth exactly face value."""
    assert fi.bond_price(c, c, 10) == pytest.approx(100.0, abs=1e-10)


@pytest.mark.parametrize("c,m,y", [(0.07, 2, 0.07), (0.07, 10, 0.065), (0.0, 10, 0.07), (0.05, 30, 0.08)])
def test_dv01_matches_finite_difference(c, m, y):
    assert fi.dv01(y, c, m) == pytest.approx(fi.dv01_bump(y, c, m), rel=1e-6)


@pytest.mark.parametrize("c,m,y", [(0.07, 10, 0.07), (0.0, 10, 0.05), (0.05, 30, 0.08)])
def test_convexity_matches_second_difference(c, m, y):
    h = 1e-4
    p0, up, dn = fi.bond_price(y, c, m), fi.bond_price(y + h, c, m), fi.bond_price(y - h, c, m)
    assert fi.convexity(y, c, m) == pytest.approx((up - 2 * p0 + dn) / (h * h * p0), rel=1e-4)


def test_coupon_bond_duration_below_maturity_and_mod_below_mac():
    mac = fi.macaulay_duration(0.07, 0.07, 10)
    assert mac < 10
    assert fi.modified_duration(0.07, 0.07, 10) == pytest.approx(mac / 1.035)


@pytest.mark.parametrize("price", [60.0, 85.0, 100.0, 104.25, 140.0])
def test_ytm_round_trip(price):
    r = fi.ytm(price, 0.07, 10)
    assert fi.bond_price(r.ytm, 0.07, 10) == pytest.approx(price, abs=1e-9)
    assert r.method == "newton"


def test_ytm_bisection_fallback():
    r = fi.ytm(96.5, 0.07, 10, max_iter=1)        # starve Newton
    assert r.method == "bisection"
    assert fi.bond_price(r.ytm, 0.07, 10) == pytest.approx(96.5, abs=1e-8)


@pytest.mark.parametrize("dy", [-0.01, 0.01])
def test_convexity_improves_the_approximation(dy):
    a = fi.price_change_approx(0.07, dy, 0.07, 30)
    assert abs(a["error_dur_cx_pct"]) < abs(a["error_duration_pct"]) / 5
    assert a["duration_pct"] < a["exact_pct"]      # duration alone always under-estimates the price


# ---------------------------------------------------------------------------
# Part 2 - yield curve
# ---------------------------------------------------------------------------
MATS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


@pytest.fixture(scope="module")
def market():
    from marketlens.data import load_fred
    row = load_fred().dropna().iloc[-1]
    return fi.par_curve(row)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_bootstrap_reprices_par_bonds(market, method):
    mats, pars = market
    c = fi.bootstrap(mats, pars, method)
    for t, p in zip(mats, pars):
        if t >= 0.5:
            assert fi.price_on_curve(c, p, t) == pytest.approx(100.0, abs=1e-10)
            assert c.par_yield(t) == pytest.approx(p, abs=1e-12)


def test_flat_par_curve_gives_flat_zero_curve():
    """If every par yield is 5% (semi-annual), every zero rate is 5% too."""
    c = fi.bootstrap(MATS, np.full(len(MATS), 0.05))
    t = np.array([0.5, 1, 2.5, 7, 15, 30])
    assert np.allclose(c.zero(t, "semi"), 0.05, atol=1e-12)
    assert np.allclose(c.forward(t[:-1], t[1:]), 2 * np.log(1.025), atol=1e-12)


def test_discount_factors_positive_and_decreasing(market):
    mats, pars = market
    c = fi.bootstrap(mats, pars, "cubic")
    d = c.df(np.linspace(0.01, 30, 600))
    assert np.all(d > 0) and np.all(np.diff(d) < 0)


def test_forward_rates_consistent_with_discount_factors(market):
    mats, pars = market
    c = fi.bootstrap(mats, pars)
    f = c.forward(3.0, 7.0)
    assert c.df(7.0) == pytest.approx(c.df(3.0) * np.exp(-f * 4.0), rel=1e-14)


# ---------------------------------------------------------------------------
# Part 3 - rates risk
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def book():
    return [fi.Position("2Y", 0.05, 2, 5e6), fi.Position("10Y", 0.05, 10, 2e6), fi.Position("30Y", 0.05, 30, 1e6)]


def test_portfolio_dv01_matches_finite_difference_bump(market, book):
    """Portfolio DV01 equals a direct reprice after a 1 bp parallel shift (to second order)."""
    mats, pars = market
    base = fi.portfolio_value(fi.bootstrap(mats, pars), book)
    up = fi.portfolio_value(fi.shifted_curve(mats, pars, 1), book)
    assert fi.portfolio_dv01(mats, pars, book) == pytest.approx(base - up, rel=2e-3)


def test_single_bond_dv01_matches_yield_dv01_on_flat_curve():
    """On a flat curve, curve DV01 of one bond = the bond's yield DV01 from Part 1."""
    pars = np.full(len(MATS), 0.05)
    pos = [fi.Position("10Y", 0.05, 10, 100)]               # face 100 -> price units
    assert fi.portfolio_dv01(MATS, pars, pos) == pytest.approx(fi.dv01(0.05, 0.05, 10), rel=1e-6)


def test_key_rate_dv01s_sum_to_parallel_dv01(market, book):
    mats, pars = market
    assert fi.key_rate_dv01(mats, pars, book).sum() == pytest.approx(
        fi.portfolio_dv01(mats, pars, book), rel=1e-3)


def test_long_bonds_lose_when_rates_rise(market, book):
    mats, pars = market
    rows = {r["Scenario"]: r for r in fi.scenario_pnl(mats, pars, book)}
    assert rows["Parallel +100 bp"]["Full reval P&L"] < 0 < rows["Parallel -100 bp"]["Full reval P&L"]
    # positive convexity: full revaluation beats the linear DV01 estimate both ways
    assert rows["Parallel +100 bp"]["Difference (convexity)"] > 0
    assert rows["Parallel -100 bp"]["Difference (convexity)"] > 0


def test_steepener_and_flattener_are_mirror_images(market):
    mats, pars = market
    s = fi.scenario_shifts(mats)
    assert np.allclose(s["Steepener (2s10s +50 bp)"], -s["Flattener (2s10s -50 bp)"])
    st = s["Steepener (2s10s +50 bp)"]
    assert st[list(mats).index(10)] - st[list(mats).index(2)] == pytest.approx(50)


# ---------------------------------------------------------------------------
# Part 4 - credit
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flat_curve():
    return fi.bootstrap(MATS, np.full(len(MATS), 0.04))


@pytest.mark.parametrize("spread", [0.005, 0.0111, 0.032, 0.08])
def test_cds_par_spread_matches_credit_triangle(flat_curve, spread):
    """Par spread ~ lambda (1 - R) when lambda = s / (1 - R)."""
    R = 0.4
    lam = fi.hazard_from_spread(spread, R)
    legs = fi.cds_legs(flat_curve, lam, R, 5.0)
    assert legs["par_spread"] == pytest.approx(lam * (1 - R), rel=0.01)


def test_cds_gap_shrinks_with_more_frequent_premiums(flat_curve):
    """The small gap vs lambda(1-R) comes from paying premiums in arrears at discrete dates."""
    lam, R = 0.05, 0.4
    gaps = [abs(fi.cds_legs(flat_curve, lam, R, 5.0, pay_freq=f)["par_spread"] - lam * (1 - R))
            for f in (1, 4, 12)]
    assert gaps[0] > gaps[1] > gaps[2]


def test_cds_mtm_zero_at_par_and_gains_when_spreads_widen(flat_curve):
    R = 0.4
    legs = fi.cds_legs(flat_curve, fi.hazard_from_spread(0.01, R), R)
    assert fi.cds_mtm(legs["par_spread"], legs) == pytest.approx(0, abs=1e-15)
    wider = fi.cds_legs(flat_curve, fi.hazard_from_spread(0.015, R), R)
    assert fi.cds_mtm(legs["par_spread"], wider) > 0


def test_survival_probabilities(flat_curve):
    t = np.linspace(0, 10, 50)
    q = fi.survival(t, 0.03)
    assert q[0] == 1 and np.all(np.diff(q) < 0)
    assert fi.survival(5, 0.03) == pytest.approx(np.exp(-0.15))


def test_bond_price_falls_as_spread_widens(flat_curve):
    prices = [fi.price_on_curve(flat_curve, 0.05, 5, spread=s) for s in np.linspace(0, 0.06, 25)]
    assert np.all(np.diff(prices) < 0)


@pytest.mark.parametrize("s", [0.0, 0.0089, 0.0111, 0.032])
def test_z_spread_round_trip(flat_curve, s):
    px = fi.price_on_curve(flat_curve, 0.06, 5, spread=s)
    assert fi.z_spread(px, flat_curve, 0.06, 5) == pytest.approx(s, abs=1e-10)
