"""Rates & Credit report.  Usage:  python scripts/run_fixed_income.py

Part 1: bond pricing, YTM, duration, convexity, DV01, +/-100 bp approximation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketlens import config, plots
from marketlens import fixed_income as fi

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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    part1()
