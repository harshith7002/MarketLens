"""Phase 2 report: prints all risk tables, saves CSVs to reports/ and charts to reports/figures/.

Usage:  python scripts/run_risk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from marketlens import config, plots, risk
from marketlens.data import load_data

pd.set_option("display.width", 140)
OUT = config.ROOT_DIR / "reports"


def main():
    prices, rets = load_data()
    OUT.mkdir(exist_ok=True)

    print("\n=== Per-asset metrics (5 years) ===")
    summ = risk.asset_summary(prices, rets)
    print(summ.to_string(formatters={
        "Ann. return": "{:.2%}".format, "Ann. volatility": "{:.2%}".format,
        "Sharpe": "{:.2f}".format, "Max drawdown": "{:.2%}".format}))
    summ.to_csv(OUT / "asset_summary.csv")

    print("\n=== Correlation matrix ===")
    corr = risk.correlation_matrix(rets)
    print(corr.round(2).to_string())
    corr.to_csv(OUT / "correlation.csv")

    rc = risk.rolling_correlation(rets, "NIFTY50", "USDINR")
    print(f"\nRolling 60d corr NIFTY50 vs USDINR: min {rc.min():.2f} ({rc.idxmin():%Y-%m-%d}), "
          f"max {rc.max():.2f} ({rc.idxmax():%Y-%m-%d}), latest {rc.iloc[-1]:.2f}")

    print(f"\n=== Portfolio: equal-weight {config.EQUITIES}, "
          f"Rs {config.PORTFOLIO_VALUE_INR:,.0f}, 1-day VaR on last {config.VAR_WINDOW} days ===")
    vt = risk.var_table(rets)
    print(vt.to_string(index=False, formatters={
        "VaR %": "{:.3%}".format, "ES %": "{:.3%}".format,
        "VaR INR": "{:,.0f}".format, "ES INR": "{:,.0f}".format}))
    vt.to_csv(OUT / "var_table.csv", index=False)

    print(f"\n=== VaR backtest: last {config.BACKTEST_DAYS} days, rolling {config.VAR_WINDOW}-day window ===")
    fc = risk.rolling_var_forecasts(rets)
    bt = risk.backtest_summary(fc)
    print(bt.to_string(index=False, formatters={
        "Expected": "{:.1f}".format, "Kupiec LR": "{:.3f}".format, "p-value": "{:.3f}".format}))
    print(f"Backtest period: {fc.index[0]:%Y-%m-%d} -> {fc.index[-1]:%Y-%m-%d}")
    fc.to_csv(OUT / "var_forecasts.csv")
    bt.to_csv(OUT / "var_backtest.csv", index=False)

    figs = [
        plots.plot_correlation(corr),
        plots.plot_rolling_corr(rc, "NIFTY50", "USDINR", corr.loc["NIFTY50", "USDINR"]),
        plots.plot_var_backtest(fc, "99"),
        plots.plot_var_backtest(fc, "95"),
        plots.plot_drawdowns(prices, ["NIFTY50", "HDFCBANK", "INFY", "GOLD"]),
    ]
    print("\nSaved figures:", *[f.name for f in figs])


if __name__ == "__main__":
    main()
