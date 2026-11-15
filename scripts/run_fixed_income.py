"""Rates & Credit report.  Usage:  python scripts/run_fixed_income.py

Part 1: bond pricing, YTM, duration, convexity, DV01, +/-100 bp approximation.
Part 2: US Treasury zero curve bootstrapped from FRED par yields, forwards.
Part 3: DV01 / key-rate DV01 of two bond books, parallel / steepener / flattener P&L.
Part 4: credit spreads (FRED ICE BofA OAS), hazard rates, survival, CDS pricing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketlens import config, plots
from marketlens import fixed_income as fi
from marketlens.data import load_fred

pd.set_option("display.width", 160)
OUT = config.ROOT_DIR / "reports"

# Illustrative bonds (inputs, not market data): semi-annual coupons, valued on
# a coupon date, all at a 7.00% yield - roughly where 10-year Indian G-Secs trade.
Y0, FREQ = 0.07, 2
BONDS = {
    "2Y 7.00%": (0.07, 2),
    "10Y 7.00%": (0.07, 10),
    "10Y zero": (0.00, 10),
    "30Y 7.00%": (0.07, 30),
}


def part1():
    print("\n=== Part 1: bond pricing and risk measures (yield 7.00%, semi-annual) ===")
    rows = {}
    for name, (c, m) in BONDS.items():
        rows[name] = {
            "Price": fi.bond_price(Y0, c, m, FREQ),
            "Macaulay dur (yrs)": fi.macaulay_duration(Y0, c, m, FREQ),
            "Modified dur": fi.modified_duration(Y0, c, m, FREQ),
            "Convexity": fi.convexity(Y0, c, m, FREQ),
            "DV01 (formula)": fi.dv01(Y0, c, m, FREQ),
            "DV01 (bump)": fi.dv01_bump(Y0, c, m, FREQ),
        }
    tbl = pd.DataFrame(rows).T
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv(OUT / "bond_measures.csv")

    print("\n--- YTM solver: price -> yield -> price round trip (10Y 7.00% bond) ---")
    for px in (85.0, 96.5, 100.0, 104.25, 120.0):
        r = fi.ytm(px, 0.07, 10, FREQ)
        back = fi.bond_price(r.ytm, 0.07, 10, FREQ)
        print(f"price {px:7.2f} -> YTM {r.ytm:.6%} ({r.method}, {r.iterations} iters) -> reprice {back:.10f}")

    print("\n--- Duration / convexity approximation vs exact repricing ---")
    rows = []
    for name, (c, m) in BONDS.items():
        for dy in (-0.02, -0.01, 0.01, 0.02):
            a = fi.price_change_approx(Y0, dy, c, m, FREQ)
            rows.append({"Bond": name, "Shift (bp)": int(round(dy / fi.BP)), "Exact %": a["exact_pct"],
                         "Duration %": a["duration_pct"], "Dur+Convexity %": a["duration_convexity_pct"],
                         "Err duration (pp)": a["error_duration_pct"] * 100,
                         "Err dur+cx (pp)": a["error_dur_cx_pct"] * 100})
    ap = pd.DataFrame(rows)
    fmt = {c: "{:.3%}".format for c in ("Exact %", "Duration %", "Dur+Convexity %")}
    fmt.update({c: "{:+.4f}".format for c in ("Err duration (pp)", "Err dur+cx (pp)")})
    print(ap.to_string(index=False, formatters=fmt))
    print("(pp = percentage points of price. Negative error = estimate too low.)")
    ap.to_csv(OUT / "bond_approximation.csv", index=False)

    ys = np.linspace(0.02, 0.12, 101)
    compare = {n: np.array([fi.bond_price(v, c, m, FREQ) for v in ys]) for n, (c, m) in BONDS.items()}
    f = plots.plot_price_yield(Y0, 0.07, 10, FREQ, lambda v: fi.bond_price(v, 0.07, 10, FREQ),
                               fi.modified_duration(Y0, 0.07, 10, FREQ), fi.convexity(Y0, 0.07, 10, FREQ),
                               compare, ys)
    print("Saved figure:", f.name)


def market_curve():
    """Latest date with both Treasury and credit data, its par curve and both bootstrapped curves."""
    fred = load_fred()
    full = fred.dropna()
    as_of = full.index[-1]
    mats, pars = fi.par_curve(full.iloc[-1])
    return fred, as_of, mats, pars


def part2():
    fred, as_of, mats, pars = market_curve()
    print(f"\n=== Part 2: US Treasury zero curve from FRED par yields, {as_of:%Y-%m-%d} ===")
    curves = {m: fi.bootstrap(mats, pars, m) for m in ("linear", "cubic")}
    tbl = pd.DataFrame({
        "Par yield %": pars * 100,
        "Zero (semi) linear %": curves["linear"].zero(mats, "semi") * 100,
        "Zero (semi) cubic %": curves["cubic"].zero(mats, "semi") * 100,
        "Discount factor": curves["linear"].df(mats),
    }, index=[f"{m:g}y" if m >= 1 else f"{round(m * 12)}m" for m in mats])
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv(OUT / "zero_curve.csv")
    for m, c in curves.items():
        err = max(abs(fi.price_on_curve(c, p, t) - 100) for t, p in zip(mats, pars) if t >= 0.5)
        print(f"{m:6s}: every quoted par bond (6m-30y) reprices to 100 within {err:.1e}")

    fw = pd.DataFrame({m: {"1y1y": c.forward(1, 2), "2y1y": c.forward(2, 3), "5y5y": c.forward(5, 10),
                           "10y10y": c.forward(10, 20), "20y10y": c.forward(20, 30)}
                       for m, c in curves.items()}) * 100
    print("\nForward rates (%, continuously compounded):")
    print(fw.to_string(float_format=lambda v: f"{v:.3f}"))
    fw.to_csv(OUT / "forward_rates.csv")
    t0 = np.arange(0, 29.5 + 1e-9, 0.5)
    for m, c in curves.items():
        f6 = c.forward(t0, t0 + 0.5)
        print(f"6-month forwards ({m}): min {f6.min():.3%}, max {f6.max():.3%}, "
              f"largest jump between neighbours {np.abs(np.diff(f6)).max() * 1e4:.0f} bp")

    hist = {}
    for label, target in (("Nov 2021", "2021-11-15"), ("Nov 2023", "2023-11-13"), ("Nov 2024", "2024-11-12")):
        d = fred.index[fred.index.get_indexer([pd.Timestamp(target)], method="nearest")[0]]
        hist[f"{label} ({d:%Y-%m-%d})"] = fi.par_curve(fred.loc[d])
    hist[f"Now ({as_of:%Y-%m-%d})"] = (mats, pars)
    for lab, (m, p) in hist.items():
        print(f"{lab}: 2y {p[4]:.2%}, 10y {p[8]:.2%}, 2s10s {(p[8] - p[4]) * 1e4:+.0f} bp")
    f = plots.plot_yield_curve(mats, pars, curves, hist, f"{as_of:%d %b %Y}")
    print("Saved figure:", f.name)
    return curves


def books(mats, pars):
    return fi.example_books(mats, pars), fi.bootstrap(mats, pars)


def part3():
    fred, as_of, mats, pars = market_curve()
    print(f"\n=== Part 3: rates risk on {as_of:%Y-%m-%d} curve ===")
    bk, curve = books(mats, pars)
    krd_all, scen_all = {}, {}
    for name, pos in bk.items():
        val = fi.portfolio_value(curve, pos)
        dv = fi.portfolio_dv01(mats, pars, pos)
        krd = fi.key_rate_dv01(mats, pars, pos)
        print(f"\n{name}")
        for p in pos:
            pv = p.face / 100 * fi.price_on_curve(curve, p.coupon, p.maturity)
            print(f"  {'LONG ' if p.face > 0 else 'SHORT'} {abs(p.face):>12,.0f}  {p.name:>3} {p.coupon:.3%} "
                  f"price {fi.price_on_curve(curve, p.coupon, p.maturity):8.4f}  value {pv:>14,.0f}  "
                  f"DV01 {fi.portfolio_dv01(mats, pars, [p]):>9,.0f}")
        print(f"  Market value {val:,.0f}   Parallel DV01 ${dv:,.0f} per bp   "
              f"(sum of key-rate DV01s ${krd.sum():,.0f})")
        rows = fi.scenario_pnl(mats, pars, pos)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
        krd_all[name], scen_all[name] = krd, rows
        pd.DataFrame(rows).to_csv(OUT / f"rates_scenarios_{name.split(':')[0].replace(' ', '_').lower()}.csv",
                                  index=False)
    kr = pd.DataFrame(krd_all, index=[f"{m:g}y" if m >= 1 else f"{round(m * 12)}m" for m in mats])
    print("\nKey-rate DV01 ($ per bp):")
    print(kr.to_string(float_format=lambda v: f"{v:,.0f}"))
    kr.to_csv(OUT / "key_rate_dv01.csv")
    f = plots.plot_rates_risk(mats, krd_all, scen_all)
    print("Saved figure:", f.name)


def part4():
    fred, as_of, mats, pars = market_curve()
    curve = fi.bootstrap(mats, pars)
    R = config.RECOVERY_RATE
    print(f"\n=== Part 4: credit on {as_of:%Y-%m-%d} (ICE BofA OAS from FRED, recovery {R:.0%} assumed) ===")
    spreads = {lab: float(fred.loc[as_of, sid]) / 100 for sid, lab in config.CREDIT_SERIES.items()}
    hist = fred[list(config.CREDIT_SERIES)].dropna()
    print(f"Spread history available {hist.index[0]:%Y-%m-%d} -> {hist.index[-1]:%Y-%m-%d}:")
    for sid, lab in config.CREDIT_SERIES.items():
        s = hist[sid]
        print(f"  {lab:14s} now {s.iloc[-1]:.2f}%  3y range {s.min():.2f}% ({s.idxmin():%Y-%m-%d}) "
              f"to {s.max():.2f}% ({s.idxmax():%Y-%m-%d})")

    rows, hazards = [], {}
    for lab, s in spreads.items():
        lam = fi.hazard_from_spread(s, R)
        legs = fi.cds_legs(curve, lam, R, 5.0)
        hazards[lab] = lam
        rows.append({"Index": lab, "Spread (bp)": s * 1e4, "Hazard lambda": lam,
                     "1y default prob": 1 - float(fi.survival(1, lam)),
                     "5y default prob": legs["default_prob"], "5y CDS RPV01": legs["rpv01"],
                     "Protection leg (per 1)": legs["protection_leg"],
                     "CDS par spread (bp)": legs["par_spread"] * 1e4,
                     "lambda(1-R) (bp)": lam * (1 - R) * 1e4,
                     "Gap (bp)": (legs["par_spread"] - lam * (1 - R)) * 1e4})
    ct = pd.DataFrame(rows).set_index("Index")
    print(ct.to_string(formatters={
        "Spread (bp)": "{:.0f}".format, "Hazard lambda": "{:.3%}".format, "1y default prob": "{:.2%}".format,
        "5y default prob": "{:.2%}".format, "5y CDS RPV01": "{:.4f}".format,
        "Protection leg (per 1)": "{:.5f}".format, "CDS par spread (bp)": "{:.2f}".format,
        "lambda(1-R) (bp)": "{:.2f}".format, "Gap (bp)": "{:+.2f}".format}))
    ct.to_csv(OUT / "credit_cds.csv")

    print("\nCDS mark-to-market, 10,000,000 notional, 5y, protection BOUGHT:")
    notional = 10_000_000
    for lab, std_cpn in (("US BBB", 0.01), ("US High Yield", 0.05)):
        legs = fi.cds_legs(curve, hazards[lab], R, 5.0)
        up = fi.cds_mtm(std_cpn, legs) * notional
        print(f"  {lab}: standard coupon {std_cpn * 1e4:.0f} bp vs par {legs['par_spread'] * 1e4:.1f} bp -> "
              f"upfront {'paid by' if up > 0 else 'received by'} buyer ${abs(up):,.0f}")
    legs0 = fi.cds_legs(curve, hazards["US BBB"], R, 5.0)
    s0 = legs0["par_spread"]
    legs1 = fi.cds_legs(curve, fi.hazard_from_spread(spreads["US BBB"] + 0.005, R), R, 5.0)
    gain = fi.cds_mtm(s0, legs1) * notional
    print(f"  BBB protection bought at par ({s0 * 1e4:.1f} bp); spreads widen 50 bp -> buyer gains ${gain:,.0f} "
          f"(~ 50 bp x RPV01 {legs1['rpv01']:.3f} x notional = ${0.005 * legs1['rpv01'] * notional:,.0f})")

    # A 5-year BBB corporate bond priced off the Treasury curve + spread
    s_bbb = spreads["US BBB"]
    cpn = round((curve.par_yield(5) + s_bbb) * 800) / 800
    print(f"\n5y BBB corporate bond, coupon {cpn:.3%} (5y Treasury par {curve.par_yield(5):.3%} + BBB spread, "
          "rounded to 1/8):")
    p_gov = fi.price_on_curve(curve, cpn, 5)
    p_bbb = fi.price_on_curve(curve, cpn, 5, spread=s_bbb)
    zs = fi.z_spread(p_bbb, curve, cpn, 5)
    print(f"  Price at zero spread (as if risk-free): {p_gov:.4f}")
    print(f"  Price at BBB spread {s_bbb * 1e4:.0f} bp: {p_bbb:.4f}   z-spread solved back from price: "
          f"{zs * 1e4:.4f} bp")
    print(f"  CS01 = {fi.cs01(curve, cpn, 5, s_bbb):.4f} per 100 face per bp "
          f"(Treasury DV01 of the same bond {fi.dv01(fi.ytm(p_bbb, cpn, 5).ytm, cpn, 5):.4f})")
    marks = {}
    for lab, s in (("IG", spreads["US IG (all)"]), ("BBB", s_bbb), ("BBB +100 bp", s_bbb + 0.01),
                   ("High Yield", spreads["US High Yield"])):
        p = fi.price_on_curve(curve, cpn, 5, spread=s)
        marks[lab] = (s * 1e4, p)
        print(f"  {lab:12s} spread {s * 1e4:6.0f} bp -> price {p:8.4f} ({(p / p_bbb - 1):+.2%} vs BBB)")
    grid = np.linspace(0, 600, 121)
    prices = np.array([fi.price_on_curve(curve, cpn, 5, spread=g / 1e4) for g in grid])
    f = plots.plot_credit(hazards, grid, prices, marks, f"5y {cpn:.3%} corporate bond")
    print("Saved figure:", f.name)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    part1()
    part2()
    part3()
    part4()
