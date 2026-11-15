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
