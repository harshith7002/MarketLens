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


# ===========================================================================
# PART 2 - Yield curve: bootstrap zero rates from par yields
# ===========================================================================
# A PAR yield is the coupon a bond needs to be priced at exactly 100. The
# Treasury publishes par yields, but to price anything we need the ZERO
# (spot) rate for every date: the yield on a single cash flow at time t.
# Bootstrapping recovers them one maturity at a time, shortest first.
# ---------------------------------------------------------------------------
from scipy.interpolate import CubicSpline  # noqa: E402


class ZeroCurve:
    """Discount curve built from nodes (t_i, D_i).

    Between nodes, ln D(t) is interpolated linearly - equivalent to a
    constant forward rate between nodes, the standard simple choice that
    keeps D(t) positive and decreasing whenever forwards are positive.
    Beyond the last node the last forward rate is held flat.
    """

    def __init__(self, times, dfs, label: str = ""):
        self.times = np.concatenate([[0.0], np.asarray(times, float)])
        self.log_df = np.concatenate([[0.0], np.log(np.asarray(dfs, float))])
        self.label = label

    def df(self, t):
        """Discount factor D(t) = value today of 1 paid at time t."""
        t = np.asarray(t, float)
        last_fwd = -(self.log_df[-1] - self.log_df[-2]) / (self.times[-1] - self.times[-2])
        out = np.interp(t, self.times, self.log_df)
        out = np.where(t > self.times[-1], self.log_df[-1] - last_fwd * (t - self.times[-1]), out)
        return np.exp(out)

    def zero(self, t, comp: str = "cont"):
        """Zero rate. comp='cont': -ln D / t.  comp='semi': 2[(1/D)^(1/(2t)) - 1]."""
        t = np.asarray(t, float)
        d = self.df(t)
        if comp == "cont":
            return -np.log(d) / t
        return 2 * (d ** (-1 / (2 * t)) - 1)

    def forward(self, t1, t2):
        """Continuously compounded forward rate from t1 to t2:
            f = ln(D(t1) / D(t2)) / (t2 - t1)
        The rate you can LOCK IN today for borrowing between t1 and t2 -
        implied by today's prices, not a forecast."""
        return np.log(self.df(t1) / self.df(t2)) / (np.asarray(t2, float) - np.asarray(t1, float))

    def par_yield(self, T: float, freq: int = 2) -> float:
        """Coupon that prices a T-year bond at 100:  c = f (1 - D_T) / sum_i D(t_i)."""
        t = np.arange(1, int(round(T * freq)) + 1) / freq
        return float(freq * (1 - self.df(T)) / np.sum(self.df(t)))


def par_curve(row, series: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(maturities in years, par yields as decimals) from one FRED row (values in %)."""
    from marketlens import config
    series = series or config.UST_SERIES
    mats = np.array(list(series.values()), float)
    pars = np.array([row[s] for s in series], float) / 100
    return mats, pars


def bootstrap(mats, pars, method: str = "linear", freq: int = 2, label: str = "") -> ZeroCurve:
    """Strip zero rates from semi-annual par yields.

    1. Short end (t <= 6 months): a bill has ONE cash flow, so its yield is
       already a zero rate:  D(t) = (1 + y/2)^(-2t).
    2. Fill in a par yield every 6 months out to 30y by interpolating the
       quoted maturities - 'linear' or 'cubic' (natural cubic spline).
    3. At each semi-annual maturity t_n a par bond satisfies
           (c/2) * sum_{i<n} D_i + (1 + c/2) * D_n = 1
       Every D_i with i < n is already known, so solve for the one unknown:
           D_n = (1 - (c/2) * sum_{i<n} D_i) / (1 + c/2)
       then move to the next maturity. That is the "bootstrap".
    """
    mats, pars = np.asarray(mats, float), np.asarray(pars, float)
    nodes: dict[float, float] = {}
    for t, y in zip(mats, pars):
        if t <= 0.5 + 1e-9:
            nodes[round(t, 10)] = (1 + y / freq) ** (-freq * t)

    grid = np.arange(1, int(round(mats.max() * freq)) + 1) / freq       # 0.5, 1.0, ..., 30.0
    if method == "linear":
        par_grid = np.interp(grid, mats, pars)
    elif method == "cubic":
        par_grid = CubicSpline(mats, pars, bc_type="natural")(grid)
    else:
        raise ValueError("method must be 'linear' or 'cubic'")

    running_sum = 0.0                                   # sum of D at earlier coupon dates
    for t, c in zip(grid, par_grid):
        d_n = (1 - c / freq * running_sum) / (1 + c / freq)
        nodes[round(t, 10)] = d_n
        running_sum += d_n

    ts = np.array(sorted(nodes))
    return ZeroCurve(ts, np.array([nodes[t] for t in ts]), label or method)


def price_on_curve(curve: ZeroCurve, coupon_rate: float, maturity: float, freq: int = 2,
                   face: float = 100.0, spread: float = 0.0) -> float:
    """Price = sum CF_k * D(t_k) * exp(-spread * t_k).

    spread = 0 gives the government (risk-free) price. spread > 0 is the
    Z-SPREAD: one extra continuously-compounded rate added on top of every
    zero rate to pay for credit (default) risk."""
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    return float(np.sum(cf * curve.df(t) * np.exp(-spread * t)))


# ===========================================================================
# PART 3 - Rates risk: DV01, key-rate DV01, curve scenarios
# ===========================================================================
@dataclass
class Position:
    name: str
    coupon: float
    maturity: float
    face: float        # notional (e.g. 5_000_000); negative = short


def example_books(mats, pars) -> dict[str, list[Position]]:
    """The two books used in the report and dashboard.

    Coupons = today's par yield rounded to 1/8% (how Treasuries are issued),
    so every bond starts close to 100.
      Book A: a plain long Treasury portfolio (2y/5y/10y/30y, $11M face).
      Book B: long $10M 10y, short 2y sized so the parallel DV01 nets to ~0
              - a 2s10s flattener: profits if the curve flattens.
    """
    par_at = dict(zip(np.round(mats, 10), pars))
    cpn = lambda m: round(par_at[m] * 800) / 800
    book_a = [Position("2Y", cpn(2), 2, 5_000_000), Position("5Y", cpn(5), 5, 3_000_000),
              Position("10Y", cpn(10), 10, 2_000_000), Position("30Y", cpn(30), 30, 1_000_000)]
    long10 = Position("10Y", cpn(10), 10, 10_000_000)
    d10 = portfolio_dv01(mats, pars, [long10])
    d2_per_unit = portfolio_dv01(mats, pars, [Position("2Y", cpn(2), 2, 1_000_000)]) / 1_000_000
    short2 = Position("2Y", cpn(2), 2, -round(d10 / d2_per_unit / 100_000) * 100_000)
    return {"Book A: long Treasury portfolio": book_a,
            "Book B: DV01-neutral 2s10s flattener": [long10, short2]}


def portfolio_value(curve: ZeroCurve, positions: list[Position], freq: int = 2) -> float:
    return sum(p.face / 100 * price_on_curve(curve, p.coupon, p.maturity, freq) for p in positions)


def shifted_curve(mats, pars, shift_bp, method: str = "linear") -> ZeroCurve:
    """Shift the quoted PAR yields by `shift_bp` (a number, or one per maturity) and
    re-bootstrap. Shocking market quotes and rebuilding is how desks run curve risk."""
    return bootstrap(mats, np.asarray(pars, float) + np.asarray(shift_bp, float) * BP, method)


def portfolio_dv01(mats, pars, positions, method: str = "linear") -> float:
    """Value change for a 1 bp PARALLEL fall in all par yields (central difference)."""
    up = portfolio_value(shifted_curve(mats, pars, 1, method), positions)
    dn = portfolio_value(shifted_curve(mats, pars, -1, method), positions)
    return (dn - up) / 2


def key_rate_dv01(mats, pars, positions, method: str = "linear") -> np.ndarray:
    """DV01 to each quoted maturity separately (bump one par yield at a time).

    Parallel DV01 says how much you lose if the WHOLE curve moves 1 bp.
    Key-rate DV01s say WHERE on the curve that risk sits, which is what
    matters for twists (steepeners, flatteners). They sum to roughly the
    parallel DV01.
    """
    out = []
    for i in range(len(mats)):
        bump = np.zeros(len(mats))
        bump[i] = 1.0
        up = portfolio_value(shifted_curve(mats, pars, bump, method), positions)
        dn = portfolio_value(shifted_curve(mats, pars, -bump, method), positions)
        out.append((dn - up) / 2)
    return np.array(out)


def scenario_shifts(mats) -> dict[str, np.ndarray]:
    """Par-yield shifts in bp at each quoted maturity.

    Parallel  : every maturity moves by the same amount.
    Steepener : 2s10s slope +50 bp. Short end (<= 2y) -25 bp, long end (>= 10y)
                +25 bp, linear in between (so it pivots at 6y).
    Flattener : the mirror image, 2s10s -50 bp.
    """
    mats = np.asarray(mats, float)
    ramp = np.clip((mats - 2) / (10 - 2), 0, 1) * 50 - 25
    return {"Parallel +100 bp": np.full(len(mats), 100.0),
            "Parallel -100 bp": np.full(len(mats), -100.0),
            "Steepener (2s10s +50 bp)": ramp,
            "Flattener (2s10s -50 bp)": -ramp}


def scenario_pnl(mats, pars, positions, method: str = "linear") -> list[dict]:
    """Full-revaluation P&L for each scenario, next to the key-rate-DV01 (linear) estimate."""
    base = portfolio_value(bootstrap(mats, pars, method), positions)
    krd = key_rate_dv01(mats, pars, positions, method)
    rows = []
    for name, shift in scenario_shifts(mats).items():
        full = portfolio_value(shifted_curve(mats, pars, shift, method), positions) - base
        approx = float(-np.sum(krd * shift))
        rows.append({"Scenario": name, "Full reval P&L": full, "Key-rate DV01 estimate": approx,
                     "Difference (convexity)": full - approx})
    return rows


# ===========================================================================
# PART 4 - Credit: spreads, hazard rates, survival, CDS
# ===========================================================================
def z_spread(price: float, curve: ZeroCurve, coupon_rate: float, maturity: float, freq: int = 2,
             face: float = 100.0) -> float:
    """Solve price_on_curve(spread=s) = price for s (Newton; price falls as s rises).

    The z-spread is the constant extra yield over the government zero curve
    the market demands to hold a corporate bond: mostly compensation for
    expected default losses, plus liquidity and risk premia."""
    t, cf = bond_cashflows(coupon_rate, maturity, freq, face)
    pv = cf * curve.df(t)
    s = 0.01
    for _ in range(100):
        disc = np.exp(-s * t)
        f = float(np.sum(pv * disc)) - price
        if abs(f) < 1e-12 * face:
            return float(s)
        s -= f / float(np.sum(-t * pv * disc))
    raise RuntimeError("z-spread did not converge")


def cs01(curve: ZeroCurve, coupon_rate: float, maturity: float, spread: float, freq: int = 2,
         face: float = 100.0) -> float:
    """Credit-spread DV01: price gain if the spread tightens by 1 bp (central difference)."""
    up = price_on_curve(curve, coupon_rate, maturity, freq, face, spread + BP)
    dn = price_on_curve(curve, coupon_rate, maturity, freq, face, spread - BP)
    return (dn - up) / 2


def hazard_from_spread(spread: float, recovery: float) -> float:
    """The 'credit triangle':  lambda ~ s / (1 - R).

    Each year the chance of default is about lambda. If default happens you
    lose (1 - R) of notional. Expected loss per year = lambda (1 - R), and a
    fair spread pays for exactly that:  s = lambda (1 - R)."""
    return spread / (1 - recovery)


def survival(t, hazard: float):
    """Q(t) = exp(-lambda t): probability of NO default up to time t
    (constant default intensity - defaults arrive like a Poisson process)."""
    return np.exp(-hazard * np.asarray(t, float))


def cds_legs(curve: ZeroCurve, hazard: float, recovery: float, maturity: float = 5.0,
             pay_freq: int = 4, steps_per_period: int = 12) -> dict:
    """Value both legs of a CDS per 1 of notional, from the protection buyer's side.

    Premium leg   : buyer pays spread s every quarter while the name survives,
                    plus premium accrued up to the default date.
                    PV = s * RPV01, where
                    RPV01 = sum_i Delta_i D(t_i) Q(t_i)              (scheduled premiums)
                          + sum_u accrued(u) D(u) [Q(u-) - Q(u)]     (accrual paid on default)
    Protection leg: seller pays (1 - R) at the default time.
                    PV = (1 - R) * sum_u D(u) [Q(u-) - Q(u)]
    PAR spread = the s that makes both legs equal = Protection / RPV01.
    (u runs over small sub-steps; D(u)[Q(u-) - Q(u)] = discounted chance of
    defaulting in that sub-step.)
    """
    pay_t = np.arange(1, int(round(maturity * pay_freq)) + 1) / pay_freq
    sched = float(np.sum((1 / pay_freq) * curve.df(pay_t) * survival(pay_t, hazard)))

    fine = np.linspace(0, maturity, int(round(maturity * pay_freq * steps_per_period)) + 1)
    q = survival(fine, hazard)
    dq = q[:-1] - q[1:]
    mid = 0.5 * (fine[:-1] + fine[1:])
    d_mid = curve.df(mid)
    accrued = mid - np.floor(mid * pay_freq + 1e-12) / pay_freq   # time since last premium date
    accrual = float(np.sum(accrued * d_mid * dq))
    protection = float((1 - recovery) * np.sum(d_mid * dq))
    rpv01 = sched + accrual
    return {"rpv01": rpv01, "rpv01_scheduled": sched, "rpv01_accrual": accrual,
            "protection_leg": protection, "par_spread": protection / rpv01,
            "default_prob": float(1 - survival(maturity, hazard))}


def cds_mtm(contract_spread: float, legs: dict) -> float:
    """Value to the protection BUYER per 1 notional:  (s_par - s_contract) * RPV01.

    Bought protection at 100 bp and the market now charges 150 bp? You pay
    50 bp a year less than fair for the same insurance: worth 0.005 * RPV01
    per 1 of notional. Standard CDS trade with a fixed coupon (100 bp IG,
    500 bp HY), so exactly this amount changes hands UPFRONT."""
    return (legs["par_spread"] - contract_spread) * legs["rpv01"]
