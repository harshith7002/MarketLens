"""Phase 3 report: Black-Scholes, Greeks, Monte Carlo convergence, implied vol, delta hedging.

Usage:  python scripts/run_options.py
Writes tables to reports/ and charts to reports/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketlens import config, options, plots
from marketlens.data import load_data, load_option_chain

pd.set_option("display.width", 140)
OUT = config.ROOT_DIR / "reports"
EX = config.EXAMPLE_OPTION


def section(title):
    print(f"\n=== {title} ===")


def black_scholes_and_greeks():
    section(f"Black-Scholes, example option {EX}")
    rows = {k: options.bs_greeks(**EX, kind=k) for k in ("call", "put")}
    tbl = pd.DataFrame(rows).T.astype(float)
    print(tbl.round(4).to_string())
    tbl.to_csv(OUT / "greeks_example.csv")

    c, p = tbl.loc["call", "price"], tbl.loc["put", "price"]
    rhs = EX["S"] - EX["K"] * np.exp(-EX["r"] * EX["T"])
    print(f"Put-call parity: C - P = {c - p:.6f}, S - K e^(-rT) = {rhs:.6f}")

    spots = np.linspace(60, 140, 161)
    greeks_by_T = {T: options.bs_greeks(spots, EX["K"], T, EX["r"], EX["sigma"], "call")
                   for T in (1.0, 0.25, 0.05)}
    return plots.plot_greeks_vs_spot(greeks_by_T, spots, EX["K"])


def monte_carlo():
    section("Monte Carlo vs Black-Scholes (ATM call, 20 seeds per path count)")
    rows = options.mc_convergence(**EX, kind="call")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={
        "mc_price_seed0": "{:.4f}".format, "bs_price": "{:.4f}".format,
        "mean_abs_error": "{:.5f}".format, "mean_abs_error_pct": "{:.3%}".format,
        "mean_std_error": "{:.5f}".format}))
    slope = np.polyfit(np.log(df["paths"]), np.log(df["mean_abs_error"]), 1)[0]
    print(f"Log-log slope of error vs paths: {slope:.3f}  (theory: -0.5)")
    p100k, se = options.mc_price(**EX, kind="call", n_paths=100_000, seed=42)
    pa, sea = options.mc_price(**EX, kind="call", n_paths=100_000, seed=42, antithetic=True)
    print(f"100k paths, seed 42: plain {p100k:.4f} (SE {se:.4f}), antithetic {pa:.4f} (SE {sea:.4f}); "
          f"BS {df['bs_price'].iloc[0]:.4f}")
    df.to_csv(OUT / "mc_convergence.csv", index=False)
    return plots.plot_mc_convergence(rows)


def implied_volatility():
    section("Implied volatility solver - round trip on known prices")
    tests = []
    for K in (60, 80, 100, 120, 160):
        for sig in (0.10, 0.30, 0.80):
            for T in (0.05, 1.0):
                price = float(options.bs_price(100, K, T, 0.05, sig, "call"))
                res = options.implied_vol(price, 100, K, T, 0.05, "call")
                tests.append({"K": K, "T": T, "true_sigma": sig, "price": price,
                              "solved_sigma": res.sigma, "method": res.method,
                              "iters": res.iterations})
    rt = pd.DataFrame(tests)
    ok = rt[rt["solved_sigma"].notna()]
    print(f"{len(rt)} cases, {len(ok)} solvable; max |error| = {np.nanmax(np.abs(ok.solved_sigma - ok.true_sigma)):.2e}")
    print("Methods used:", rt["method"].value_counts().to_dict(), f"max iterations {ok.iters.max()}")
    print("Rejected cases (time value ~ 0, sigma not identifiable):")
    print(rt[rt["solved_sigma"].isna()][["K", "T", "true_sigma", "price", "method"]].to_string(index=False))
    rt.to_csv(OUT / "iv_round_trip.csv", index=False)

    section("Implied volatility - real SPY option chain (cached snapshot)")
    chain = load_option_chain()
    expiry, qd = chain["expiry"].iloc[0], chain["quote_date"].iloc[0]
    T = (pd.Timestamp(expiry) - pd.Timestamp(qd)).days / 365
    r = config.USD_RISK_FREE_RATE
    fwd = options.implied_forward(chain, T, r)
    print(f"Quotes {qd}, expiry {expiry}, T = {T:.4f} yr, spot {fwd['spot']:.2f}, r = {r:.2%} (assumed)")
    print(f"Implied forward {fwd['forward']:.2f} (sd across {fwd['n_strikes']} strikes {fwd['forward_std']:.3f}) "
          f"-> implied carry q = {fwd['q']:.2%}")
    print(f"Parity slope d(C-P)/dK = {fwd['parity_slope']:.4f} vs European theory {fwd['parity_slope_theory']:.4f}")
    smile = options.iv_smile(chain, T, r, fwd["q"])
    atm = smile.iloc[(smile["moneyness"] - 1).abs().argmin()]
    lo, hi = smile.iloc[0], smile.iloc[-1]
    print(f"{len(smile)} liquid OTM options solved; methods {smile['iv_method'].value_counts().to_dict()}, "
          f"max iterations {smile['iv_iters'].max()}")
    print(f"ATM IV {atm.iv:.2%} at K={atm.strike:.0f};  K/F={lo.moneyness:.2f}: {lo.iv:.2%};  "
          f"min IV {smile.iv.min():.2%} at K/F={smile.loc[smile.iv.idxmin(), 'moneyness']:.2f}")
    print(f"Mean |our IV - Yahoo IV| = {(smile.iv - smile.impliedVolatility).abs().mean() * 100:.2f} vol points")
    smile.to_csv(OUT / "iv_smile_spy.csv", index=False)
    return plots.plot_iv_smile(smile, expiry, qd, config.OPTION_TICKER)


def hedging_simulation():
    section("Delta hedging - simulated GBM paths, 3-month ATM call, sigma 20%")
    rng = np.random.default_rng(config.SEED)
    n_days, n_paths, S0, r, sig = 60, 20_000, 100.0, EX["r"], 0.20
    T = n_days / 252
    paths = options.gbm_paths(S0, T, r, sig, n_days, n_paths, rng, mu=0.12)  # real-world drift 12%
    results, rows = {}, []
    for label, step in (("Daily", 1), ("Weekly", 5), ("Monthly", 20)):
        h = options.delta_hedge(paths[::step], 100, T, r, sig)
        results[label] = h["pnl"]
        rows.append({"Scenario": f"{label} hedge", "Rebalances": n_days // step,
                     "Mean P&L": h["pnl"].mean(), "Std P&L": h["pnl"].std(),
                     "Std / premium": h["pnl"].std() / h["premium"].mean()})
    premium = float(options.bs_price(S0, 100, T, r, sig))

    h = options.delta_hedge(paths, 100, T, r, sig, cost_bps=5)
    rows.append({"Scenario": "Daily hedge + 5 bp costs", "Rebalances": n_days,
                 "Mean P&L": h["pnl"].mean(), "Std P&L": h["pnl"].std(),
                 "Std / premium": h["pnl"].std() / premium})
    hot = options.gbm_paths(S0, T, r, 0.25, n_days, n_paths, np.random.default_rng(7), mu=0.12)
    h = options.delta_hedge(hot, 100, T, r, sig)
    rows.append({"Scenario": "Daily, realised vol 25% > sold at 20%", "Rebalances": n_days,
                 "Mean P&L": h["pnl"].mean(), "Std P&L": h["pnl"].std(),
                 "Std / premium": h["pnl"].std() / premium})
    tbl = pd.DataFrame(rows)
    print(f"Premium received: {premium:.4f}; {n_paths:,} paths; real-world drift 12% (not r)")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    vega_diff = float(options.bs_price(S0, 100, T, r, 0.25) - premium)
    print(f"For reference, BS price at 25% minus at 20% = {vega_diff:.4f}")
    tbl.to_csv(OUT / "hedge_simulation.csv", index=False)
    return plots.plot_hedge_pnl_hist(results, premium)


def hedging_real_nifty():
    section(f"Delta hedging - REAL NIFTY 50 path, last {config.HEDGE_DAYS} trading days")
    prices, rets = load_data()
    path = prices["NIFTY50"].iloc[-(config.HEDGE_DAYS + 1):]
    start = path.index[0]
    # Volatility used to sell & hedge = trailing 1-year realised vol, known at the start (no look-ahead)
    past = rets.loc[:start, "NIFTY50"].iloc[-config.VAR_WINDOW:]
    sig = float(past.std() * np.sqrt(config.TRADING_DAYS))
    realised = float(np.log(path).diff().dropna().std() * np.sqrt(config.TRADING_DAYS))
    T, r, K = config.HEDGE_DAYS / config.TRADING_DAYS, config.RISK_FREE_RATE, float(path.iloc[0])

    detail = pd.DataFrame(options.delta_hedge_path_detail(path.to_numpy(), K, T, r, sig), index=path.index)
    # Gamma-theta approximation of each day's P&L for the hedged SHORT call:
    #   dPnL ~ 0.5 * Gamma * S^2 * (sigma^2 dt - (dS/S)^2)
    dt = T / config.HEDGE_DAYS
    ret = detail["S"].pct_change()
    daily = 0.5 * detail["gamma"].shift(1) * detail["S"].shift(1) ** 2 * (sig**2 * dt - ret**2)
    detail["gamma_theta_approx"] = daily.fillna(0).cumsum()

    final = detail.iloc[-1]
    premium = float(options.bs_price(K, K, T, r, sig))
    print(f"Period {start:%Y-%m-%d} -> {path.index[-1]:%Y-%m-%d}; ATM strike K = {K:,.2f}")
    print(f"Sold & hedged at trailing 1y realised vol {sig:.2%}; vol actually realised {realised:.2%}")
    print(f"Premium received {premium:,.2f} pts; final NIFTY {final.S:,.2f}; payoff {max(final.S - K, 0):,.2f} pts")
    print(f"Final hedged P&L {final.book_pnl:,.2f} pts ({final.book_pnl / premium:.1%} of premium); "
          f"gamma-theta approximation {final.gamma_theta_approx:,.2f} pts")
    print(f"Unhedged short call P&L would be {premium * np.exp(r * T) - max(final.S - K, 0):,.2f} pts")
    detail.to_csv(OUT / "delta_hedge_nifty.csv")
    title = f"Short 3-month ATM NIFTY call, delta-hedged daily ({start:%d %b %Y} - {path.index[-1]:%d %b %Y})"
    return plots.plot_delta_hedge_real(detail, K, title)


def main():
    OUT.mkdir(exist_ok=True)
    figs = [black_scholes_and_greeks(), monte_carlo(), implied_volatility(),
            hedging_simulation(), hedging_real_nifty()]
    print("\nSaved figures:", *[f.name for f in figs])


if __name__ == "__main__":
    main()
