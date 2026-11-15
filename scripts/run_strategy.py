"""Phase 4 report: cointegration tests and pairs-trading backtest.

Usage:  python scripts/run_strategy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketlens import config, plots, strategy
from marketlens.data import load_data

pd.set_option("display.width", 160)
OUT = config.ROOT_DIR / "reports"
PCT = "{:.2%}".format


def fmt_metrics(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    for c in ("Total return", "CAGR", "Volatility", "Max drawdown", "Win rate", "Avg trade"):
        if c in df:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else PCT(v))
    for c in ("Sharpe", "Avg holding days"):
        if c in df:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    if "Trades" in df:
        df["Trades"] = df["Trades"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    return df


def evaluate(bt, start):
    """Metrics on the out-of-sample period only (from `start`)."""
    r = bt.loc[start:, "net_ret"]
    trades = strategy.trade_log(bt)
    trades = trades[trades["entry"] >= start] if len(trades) else trades
    m = strategy.metrics(r, rf=0.0, trades=trades)
    m["Time in market"] = float((bt.loc[start:, "position"] != 0).mean())
    return m, trades


def main():
    prices, _ = load_data()
    OUT.mkdir(exist_ok=True)
    Y, X = config.PAIR
    form = prices.iloc[: config.FORMATION_DAYS]
    start = prices.index[config.FORMATION_DAYS]

    print(f"\n=== Cointegration screen - formation period {form.index[0]:%Y-%m-%d} -> {form.index[-1]:%Y-%m-%d} ===")
    screen = strategy.pair_screen(form, config.EQUITIES)
    print(screen.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    screen.to_csv(OUT / "pairs_screen_formation.csv", index=False)
    print(f"Note: with {len(screen)} pairs tested, ~{len(screen) * 0.05:.1f} would pass p<0.05 by pure chance.")

    print(f"\n=== Engle-Granger: {Y} vs {X} ===")
    rows = {}
    for label, df in (("Formation (first 2y)", form), ("Trading (last 3y)", prices.loc[start:]),
                      ("Full 5 years", prices)):
        rows[label] = strategy.engle_granger(df[Y], df[X])
    eg = pd.DataFrame(rows).T
    print(eg[["beta", "t_stat", "crit_5pct", "p_value", "half_life_days", "n_obs"]].to_string(float_format=lambda v: f"{v:.3f}"))
    eg.to_csv(OUT / "pairs_engle_granger.csv")

    print(f"\n=== Backtest on the out-of-sample period {start:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d} ===")
    print(f"Rules: entry |z|>{config.Z_ENTRY}, exit |z|<{config.Z_EXIT}, stop |z|>{config.Z_STOP}, "
          f"z window {config.Z_WINDOW}d, beta window {config.BETA_WINDOW}d, "
          f"execution lag {config.EXEC_LAG}d, costs {config.COST_BPS} bp per leg")
    results, curves, all_trades = {}, {}, {}
    for method, label in (("ols", "Pairs - rolling EG hedge ratio"), ("ratio", "Pairs - 1:1 log-ratio")):
        bt = strategy.backtest(prices[Y], prices[X], method=method)
        m, trades = evaluate(bt, start)
        results[label] = m
        curves[label] = (1 + bt.loc[start:, "net_ret"]).cumprod()
        all_trades[method] = trades
        bt.to_csv(OUT / f"pairs_backtest_{method}.csv")
        trades.to_csv(OUT / f"pairs_trades_{method}.csv", index=False)
        plots.plot_pair_signals(bt.loc[start:], Y, X, config.Z_ENTRY, config.Z_EXIT, method)

    simple = prices.loc[start:].pct_change().fillna(0)
    bh_pair = (simple[[Y, X]] + 1).cumprod().mean(axis=1)  # Rs 0.5 in each, never rebalanced
    bench = {f"Buy & hold 50/50 {Y}+{X}": bh_pair.pct_change().fillna(0),
             "Buy & hold NIFTY 50": simple["NIFTY50"]}
    for label, r in bench.items():
        results[label] = strategy.metrics(r, rf=0.0)
        results[label]["Time in market"] = 1.0
        curves[label] = (1 + r).cumprod()
    tbl = fmt_metrics(results)
    tbl["Time in market"] = [PCT(v) for v in pd.DataFrame(results).T["Time in market"]]
    print(tbl.to_string())
    pd.DataFrame(results).T.to_csv(OUT / "pairs_performance.csv")
    print("(Sharpe here uses rf = 0 for every line, so the long-short and long-only numbers are comparable;"
          f" with rf = {config.RISK_FREE_RATE:.0%} every Sharpe would be lower.)")

    print("\n=== Trade log - rolling EG hedge ratio ===")
    t = all_trades["ols"].copy()
    if len(t):
        t["return"] = t["return"].map(PCT)
        print(t.to_string(index=False))

    print("\n=== Transaction-cost sensitivity (rolling EG method, out-of-sample) ===")
    sens = {}
    for c in (0, 5, 10, 20):
        bt = strategy.backtest(prices[Y], prices[X], method="ols", cost_bps=c)
        sens[f"{c} bp"], _ = evaluate(bt, start)
    print(fmt_metrics(sens)[["Total return", "CAGR", "Sharpe", "Max drawdown", "Win rate", "Trades"]].to_string())

    print("\n=== Robustness: same rules on TCS / INFY (most correlated pair, 0.73) ===")
    rob = {}
    for method in ("ols", "ratio"):
        bt = strategy.backtest(prices["TCS"], prices["INFY"], method=method)
        rob[f"TCS/INFY {method}"], _ = evaluate(bt, start)
    print(fmt_metrics(rob)[["Total return", "CAGR", "Sharpe", "Max drawdown", "Win rate", "Trades"]].to_string())

    f = plots.plot_equity_curves(curves, f"Out-of-sample equity curves, {start:%b %Y} - {prices.index[-1]:%b %Y} (after costs)")
    print("\nSaved figures: pairs_signals_ols.png pairs_signals_ratio.png", f.name)


if __name__ == "__main__":
    main()
