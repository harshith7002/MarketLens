"""Rates & Credit - fixed-income analytics.

Part 1 (this file so far): fixed-coupon bond pricing
------------------------------------------------------
bond_cashflows ......... coupon dates and amounts
bond_price ............. price from a yield (yield-to-maturity)
ytm .................... yield from a price (Newton-Raphson, bisection fallback)
macaulay_duration ...... cash-flow-weighted average time
modified_duration ...... % price change per 1.00 change in yield
convexity .............. curvature of the price-yield curve
dv01 ................... rupee/dollar price change for a 1 bp yield move
price_change_approx .... duration (+ convexity) estimate vs exact repricing

Conventions
-----------
* Prices are per 100 face value (market convention, "clean = dirty" because
  we value on a coupon date, so there is no accrued interest).
* Yields are annual, compounded `freq` times a year (freq=2 = semi-annual,
  the convention for both Indian G-Secs and US Treasuries).
* Times are in years. Δy in decimals (0.01 = 100 bp = 1%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BP = 1e-4  # one basis point


# ===========================================================================
# 1. Cash flows and price
# ===========================================================================
def bond_cashflows(coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0):
    """Times (years) and amounts of every cash flow.

    A 7% semi-annual 10-year bond pays 3.5 every 6 months for 20 periods,
    plus the 100 face value with the last coupon.
    """
    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity must be at least one coupon period")
    times = np.arange(1, n + 1) / freq
    cfs = np.full(n, face * coupon_rate / freq)
    cfs[-1] += face
    return times, cfs


def bond_price(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0) -> float:
    """Price = sum of cash flows discounted at the yield.

        P = sum_k  CF_k / (1 + y/f)^(f * t_k)

    The ONE rate y that discounts all cash flows back to the market price is
    the yield-to-maturity. Higher yield -> bigger denominator -> lower price:
    bond prices and yields always move in opposite directions.
    """
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    return float(np.sum(cf / (1 + y / freq) ** (freq * t)))


# ===========================================================================
# 2. Yield from price
# ===========================================================================
@dataclass
class YTMResult:
    ytm: float
    iterations: int
    method: str


def ytm(price: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0,
        tol: float = 1e-12, max_iter: int = 100) -> YTMResult:
    """Solve bond_price(y) = price for y.

    Newton-Raphson:  y <- y - (P(y) - price) / P'(y)
    where P'(y) = dP/dy = -modified_duration * P  (the slope of the
    price-yield curve). The curve is smooth and convex, so Newton converges
    in a handful of steps from the coupon rate. Bisection on [-0.99f, 1] is
    the safety net (price is strictly decreasing in y, so the root is unique).
    """
    if price <= 0:
        raise ValueError("price must be positive")
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    y = coupon_rate if coupon_rate > 0 else 0.05
    for i in range(1, max_iter + 1):
        disc = (1 + y / freq) ** (-freq * t)
        p = float(np.sum(cf * disc))
        diff = p - price
        if abs(diff) < tol * face:
            return YTMResult(float(y), i, "newton")
        dp_dy = float(np.sum(-t * cf * disc / (1 + y / freq)))
        if dp_dy == 0:
            break
        y_new = y - diff / dp_dy
        if not (-0.99 * freq < y_new < 1.0):
            break
        y = y_new

    lo, hi = -0.99 * freq + 1e-9, 1.0       # price(lo) is huge, price(hi) is small
    for j in range(1, 300):
        mid = 0.5 * (lo + hi)
        diff = bond_price(mid, coupon_rate, maturity, freq, face) - price
        if abs(diff) < tol * face or hi - lo < 1e-14:
            return YTMResult(mid, j, "bisection")
        if diff > 0:   # price too high -> yield too low
            lo = mid
        else:
            hi = mid
    return YTMResult(0.5 * (lo + hi), j, "bisection")


# ===========================================================================
# 3. Risk measures
# ===========================================================================
def macaulay_duration(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0) -> float:
    """D_mac = sum(t_k * PV_k) / P   (years)

    The present-value-weighted average time until you get your money back.
    A zero-coupon bond pays everything at maturity, so D_mac = maturity.
    Coupons pull money forward, so a coupon bond's D_mac < maturity.
    """
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    pv = cf / (1 + y / freq) ** (freq * t)
    return float(np.sum(t * pv) / np.sum(pv))


def modified_duration(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0) -> float:
    """D_mod = D_mac / (1 + y/f) = -(1/P) dP/dy

    The practical one: a bond with D_mod = 7 loses about 7% of its value if
    its yield rises by 1 percentage point (100 bp).
    """
    return macaulay_duration(y, coupon_rate, maturity, freq, face) / (1 + y / freq)


def convexity(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0) -> float:
    """C = (1/P) d2P/dy2 = sum( CF_k * t_k * (t_k + 1/f) / (1+y/f)^(f t_k + 2) ) / P

    Duration is the slope of the price-yield curve; convexity is how much it
    bends. Because the curve bends upward, a straight-line (duration-only)
    estimate always UNDER-states the price: too pessimistic when yields rise,
    not optimistic enough when they fall. Long bonds have much more convexity.
    """
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    g = 1 + y / freq
    p = np.sum(cf * g ** (-freq * t))
    return float(np.sum(cf * t * (t + 1 / freq) * g ** (-freq * t - 2)) / p)


def dv01(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0) -> float:
    """Dollar value of 1 bp: DV01 = D_mod * P * 0.0001  (price units per 100 face).

    Traders size and hedge rates positions in DV01, not in notional: two
    positions with equal and opposite DV01 are hedged against small parallel
    moves even if their face values are very different.
    """
    return modified_duration(y, coupon_rate, maturity, freq, face) * bond_price(y, coupon_rate, maturity, freq, face) * BP


def dv01_bump(y: float, coupon_rate: float, maturity: float, freq: int = 2, face: float = 100.0,
              bump: float = BP) -> float:
    """DV01 by central finite difference: [P(y - 1bp) - P(y + 1bp)] / 2.

    No formula needed - just reprice. This is how risk systems compute DV01
    for anything, including instruments with no closed-form duration.
    """
    up = bond_price(y + bump, coupon_rate, maturity, freq, face)
    dn = bond_price(y - bump, coupon_rate, maturity, freq, face)
    return (dn - up) / 2 * (BP / bump)


def price_change_approx(y: float, dy: float, coupon_rate: float, maturity: float, freq: int = 2,
                        face: float = 100.0) -> dict:
    """Compare exact repricing with the Taylor-series estimates.

        dP/P ~ -D_mod * dy                       (duration only, a straight line)
        dP/P ~ -D_mod * dy + 0.5 * C * dy^2      (duration + convexity, a parabola)
    """
    p0 = bond_price(y, coupon_rate, maturity, freq, face)
    p1 = bond_price(y + dy, coupon_rate, maturity, freq, face)
    dm = modified_duration(y, coupon_rate, maturity, freq, face)
    cx = convexity(y, coupon_rate, maturity, freq, face)
    exact = (p1 - p0) / p0
    dur = -dm * dy
    dur_cx = dur + 0.5 * cx * dy**2
    return {"price": p0, "new_price": p1, "exact_pct": exact,
            "duration_pct": dur, "duration_convexity_pct": dur_cx,
            "error_duration_pct": dur - exact, "error_dur_cx_pct": dur_cx - exact}
