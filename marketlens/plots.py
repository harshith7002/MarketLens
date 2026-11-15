"""Static matplotlib charts saved to reports/figures (used in the README).

The Streamlit app (Phase 6) draws its own interactive plotly versions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no window needed - we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from marketlens import config

# Validated categorical order (blue, orange, aqua, ...) and chart ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BREACH = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "lines.linewidth": 1.6,
})


def _save(fig, name: str):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title("Correlation of daily log returns (5 years)")
    return _save(fig, "correlation_matrix.png")


def plot_rolling_corr(series: pd.Series, a: str, b: str, full_corr: float):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(series.index, series.values, color=SERIES[0])
    ax.axhline(full_corr, color=INK_2, ls="--", lw=1)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.text(series.index[0], full_corr, f" 5-yr average {full_corr:.2f}", va="bottom", color=INK_2, fontsize=9)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Rolling {config.ROLLING_CORR_WINDOW}-day correlation: {a} vs {b}")
    return _save(fig, f"rolling_corr_{a}_{b}.png")


def plot_var_backtest(fc: pd.DataFrame, level_tag: str = "99"):
    """Realised daily P&L vs the negative of each method's VaR forecast."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(fc.index, fc["Realised"] * 100, color="#b9b8b2", width=1.0, label="Realised portfolio return")
    for i, m in enumerate(["Historical", "Parametric", "Monte Carlo"]):
        ax.plot(fc.index, -fc[f"{m}_{level_tag}"] * 100, color=SERIES[i], label=f"-VaR {level_tag}% {m}")
    hit = fc["Realised"] < -fc[f"Historical_{level_tag}"]
    ax.scatter(fc.index[hit], fc["Realised"][hit] * 100, s=40, color=BREACH, zorder=5,
               edgecolor=SURFACE, linewidth=1.5, label="Breach (historical)")
    ax.set_ylabel("Daily return (%)")
    ax.set_title(f"VaR {level_tag}% backtest - last {len(fc)} trading days (out-of-sample)")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    return _save(fig, f"var_backtest_{level_tag}.png")


def plot_drawdowns(prices: pd.DataFrame, cols: list[str]):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, c in enumerate(cols):
        dd = prices[c] / prices[c].cummax() - 1
        ax.plot(dd.index, dd * 100, color=SERIES[i], label=c, lw=1.4)
    ax.set_ylabel("Drawdown from peak (%)")
    ax.set_title("Drawdowns")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "drawdowns.png")


# ---------------------------------------------------------------------------
# Phase 3 charts
# ---------------------------------------------------------------------------
def plot_mc_convergence(rows: list[dict]):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(df["paths"], df["mean_abs_error"], "o-", color=SERIES[0], ms=6,
              label="Mean |MC - Black-Scholes| (20 seeds)")
    ax.loglog(df["paths"], df["mean_std_error"], "s--", color=SERIES[1], ms=5,
              label="Average standard error")
    ref = df["mean_abs_error"].iloc[0] * np.sqrt(df["paths"].iloc[0] / df["paths"])
    ax.loglog(df["paths"], ref, color=INK_2, lw=1, ls=":", label="1/sqrt(N) reference")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Pricing error (log scale)")
    ax.set_title("Monte Carlo converges to Black-Scholes at rate 1/sqrt(N)")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "mc_convergence.png")


def plot_greeks_vs_spot(greeks_by_T: dict, spots: np.ndarray, K: float):
    names = ["price", "delta", "gamma", "vega", "theta", "rho"]
    titles = ["Call price", "Delta", "Gamma", "Vega (per vol pt)", "Theta (per day)", "Rho (per 1%)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, n, t in zip(axes.flat, names, titles):
        for i, (T, g) in enumerate(greeks_by_T.items()):
            ax.plot(spots, g[n], color=SERIES[i], label=f"T = {T:g} yr")
        ax.axvline(K, color=INK_2, lw=0.8, ls="--")
        ax.set_title(t, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Spot price (strike = 100)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Black-Scholes call Greeks vs spot, for three maturities", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "greeks_vs_spot.png")


def plot_iv_smile(smile: pd.DataFrame, expiry: str, quote_date: str, ticker: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, kind in enumerate(["put", "call"]):
        d = smile[smile["type"] == kind]
        ax.plot(d["moneyness"], d["iv"] * 100, "o", ms=4, color=SERIES[i],
                label=f"OTM {kind}s (our solver)")
    ax.plot(smile["moneyness"], smile["impliedVolatility"] * 100, color=INK_2, lw=1, ls="--",
            label="Yahoo's quoted IV")
    ax.axvline(1.0, color=INK_2, lw=0.8)
    ax.set_xlabel("Moneyness  K / Forward")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{ticker} implied-volatility skew - expiry {expiry}, quotes {quote_date}")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "iv_smile.png")


def plot_delta_hedge_real(detail: pd.DataFrame, K: float, title: str):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1.2, 1.5]})
    a1.plot(detail.index, detail["S"], color=SERIES[0])
    a1.axhline(K, color=INK_2, ls="--", lw=1)
    a1.text(detail.index[0], K, " strike", va="bottom", color=INK_2, fontsize=9)
    a1.set_ylabel("NIFTY 50")
    a1.set_title(title)
    a2.plot(detail.index, detail["delta"], color=SERIES[2])
    a2.set_ylabel("Hedge delta")
    a2.set_ylim(-0.02, 1.02)
    a3.plot(detail.index, detail["book_pnl"], color=SERIES[1], label="Hedged book P&L (actual)")
    a3.plot(detail.index, detail["gamma_theta_approx"], color=INK_2, ls="--", lw=1.2,
            label="Gamma-theta approximation")
    a3.axhline(0, color=INK_2, lw=0.8)
    a3.set_ylabel("P&L (index points)")
    a3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "delta_hedge_nifty.png")


def plot_hedge_pnl_hist(results: dict[str, np.ndarray], premium: float):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    allv = np.concatenate(list(results.values()))
    bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 70)
    for i, (label, pnl) in enumerate(results.items()):
        ax.hist(pnl, bins=bins, color=SERIES[i], alpha=0.55, label=f"{label} (sd {pnl.std():.2f})")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(f"Hedging P&L at expiry per option (premium = {premium:.2f})")
    ax.set_ylabel("Number of paths")
    ax.set_title("Short call, delta-hedged: more frequent hedging = smaller error")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "hedge_pnl_distribution.png")


# ---------------------------------------------------------------------------
# Phase 4 charts
# ---------------------------------------------------------------------------
def plot_pair_signals(bt: pd.DataFrame, y_name: str, x_name: str, entry: float, exit: float, tag: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1.5]})
    a1.plot(bt.index, bt["Y"] / bt["Y"].iloc[0] * 100, color=SERIES[0], label=y_name)
    a1.plot(bt.index, bt["X"] / bt["X"].iloc[0] * 100, color=SERIES[1], label=x_name)
    a1.set_ylabel("Price (rebased to 100)")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title(f"Pairs trade {y_name} / {x_name} - spread z-score and positions ({tag})")
    a2.plot(bt.index, bt["z"], color=INK_2, lw=1)
    for lvl, ls in ((entry, "--"), (-entry, "--"), (exit, ":"), (-exit, ":")):
        a2.axhline(lvl, color=INK_2, ls=ls, lw=0.9)
    long_, short_ = bt["position"] > 0, bt["position"] < 0
    a2.fill_between(bt.index, -4.5, 4.5, where=long_, color=SERIES[2], alpha=0.18, lw=0, label="Long spread")
    a2.fill_between(bt.index, -4.5, 4.5, where=short_, color=SERIES[7], alpha=0.15, lw=0, label="Short spread")
    a2.set_ylim(-4.5, 4.5)
    a2.set_ylabel("Spread z-score")
    a2.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    return _save(fig, f"pairs_signals_{tag}.png")


def plot_equity_curves(curves: dict[str, pd.Series], title: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (label, eq) in enumerate(curves.items()):
        ax.plot(eq.index, eq.values, color=SERIES[i], label=f"{label}  (ends {eq.values[-1]:.2f})", lw=1.6)
    ax.axhline(1, color=INK_2, lw=0.8)
    ax.set_ylabel("Growth of Rs 1")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, "pairs_equity_curves.png")


# ---------------------------------------------------------------------------
# Rates & Credit charts
# ---------------------------------------------------------------------------
def plot_price_yield(y0: float, coupon: float, maturity: float, freq: int, price_fn, dmod: float,
                     cx: float, compare: dict[str, np.ndarray], yields_cmp: np.ndarray):
    """Left: exact price-yield curve vs duration line vs duration+convexity parabola.
    Right: price-yield curves of bonds with different maturities (convexity grows with maturity)."""
    ys = np.linspace(y0 - 0.04, y0 + 0.04, 161)
    p0 = price_fn(y0)
    exact = np.array([price_fn(v) for v in ys])
    dur = p0 * (1 - dmod * (ys - y0))
    dcx = p0 * (1 - dmod * (ys - y0) + 0.5 * cx * (ys - y0) ** 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(ys * 100, exact, color=SERIES[0], lw=2.2, label="Exact price")
    a1.plot(ys * 100, dur, color=SERIES[1], ls="--", label="Duration only (tangent line)")
    a1.plot(ys * 100, dcx, color=SERIES[2], ls=":", lw=2, label="Duration + convexity")
    for dy in (-0.01, 0.01):
        a1.axvline((y0 + dy) * 100, color=GRID, lw=1)
    a1.scatter([y0 * 100], [p0], color=INK, s=30, zorder=5)
    a1.set_xlabel("Yield (%)")
    a1.set_ylabel("Price per 100 face")
    a1.set_title(f"{maturity:g}-year {coupon:.2%} bond: price vs yield")
    a1.legend(frameon=False, fontsize=9)
    for i, (lab, prices) in enumerate(compare.items()):
        a2.plot(yields_cmp * 100, prices, color=SERIES[i], lw=2, label=lab)
    a2.set_xlabel("Yield (%)")
    a2.set_ylabel("Price per 100 face")
    a2.set_title("Longer bonds: steeper (duration) and more curved (convexity)")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, "bond_price_yield.png")


def plot_yield_curve(mats, pars, curves: dict, history: dict[str, tuple], as_of: str):
    """Left: quoted par yields + bootstrapped zero curves. Middle: 6-month forward rates
    (linear vs cubic interpolation). Right: par curves on earlier dates."""
    t = np.linspace(0.25, 30, 300)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6))
    a1.scatter(mats, pars * 100, color=INK, s=28, zorder=5, label="Quoted par yields (FRED)")
    for i, (lab, c) in enumerate(curves.items()):
        a1.plot(t, c.zero(t, "semi") * 100, color=SERIES[i], lw=1.8, label=f"Zero curve ({lab})")
    a1.set_xlabel("Maturity (years)")
    a1.set_ylabel("Rate (%)")
    a1.set_title(f"US Treasury curve, {as_of}")
    a1.legend(frameon=False, fontsize=8)
    t0 = np.arange(0, 29.5 + 1e-9, 0.5)
    for i, (lab, c) in enumerate(curves.items()):
        a2.step(t0 + 0.5, c.forward(t0, t0 + 0.5) * 100, where="pre", color=SERIES[i], lw=1.6,
                label=f"6-month forwards ({lab})")
    a2.set_xlabel("Forward start + 6m (years)")
    a2.set_ylabel("Rate (%, cont. comp.)")
    a2.set_title("Forward rates: interpolation choice matters")
    a2.legend(frameon=False, fontsize=8)
    for i, (lab, (m, p)) in enumerate(history.items()):
        a3.plot(m, p * 100, "o-", color=SERIES[i], ms=4, lw=1.6, label=lab)
    a3.set_xlabel("Maturity (years)")
    a3.set_ylabel("Par yield (%)")
    a3.set_title("How the curve has moved")
    a3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "yield_curve.png")


def plot_rates_risk(mats, krd: dict[str, np.ndarray], scen: dict[str, list[dict]]):
    """Left: key-rate DV01 by maturity for each book. Right: scenario P&L for each book."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [f"{m:g}y" if m >= 1 else f"{int(round(m * 12))}m" for m in mats]
    x = np.arange(len(mats))
    w = 0.38
    for i, (book, v) in enumerate(krd.items()):
        a1.bar(x + (i - 0.5) * w, v, width=w, color=SERIES[i], label=book)
    a1.axhline(0, color=INK_2, lw=0.8)
    a1.set_xticks(x, labels)
    a1.set_ylabel("Key-rate DV01 ($ per 1 bp)")
    a1.set_title("Where the rates risk sits")
    a1.legend(frameon=False, fontsize=8)
    names = [r["Scenario"] for r in next(iter(scen.values()))]
    x2 = np.arange(len(names))
    for i, (book, rows) in enumerate(scen.items()):
        a2.barh(x2 + (i - 0.5) * w, [r["Full reval P&L"] / 1e3 for r in rows], height=w,
                color=SERIES[i], label=book)
    a2.axvline(0, color=INK_2, lw=0.8)
    a2.set_yticks(x2, names)
    a2.invert_yaxis()
    a2.set_xlabel("P&L ($ thousands, full revaluation)")
    a2.set_title("Scenario P&L")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "rates_scenarios.png")


def plot_credit(hazards: dict[str, float], spreads_bp: np.ndarray, prices: np.ndarray,
                marks: dict[str, tuple], bond_label: str):
    """Left: survival probability curves. Right: corporate bond price vs credit spread."""
    t = np.linspace(0, 10, 201)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (lab, lam) in enumerate(hazards.items()):
        q = np.exp(-lam * t)
        a1.plot(t, q * 100, color=SERIES[i], lw=2, label=f"{lab} (lambda = {lam:.2%})")
    a1.set_xlabel("Years")
    a1.set_ylabel("Survival probability (%)")
    a1.set_title("Q(t) = exp(-lambda t), lambda = spread / (1 - R), R = 40%")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(spreads_bp, prices, color=SERIES[0], lw=2)
    for i, (lab, (s, p)) in enumerate(marks.items()):
        a2.scatter([s], [p], color=SERIES[i + 1], s=45, zorder=5, edgecolor=SURFACE, linewidth=1.5)
        # alternate labels above-right / below-left so close points (IG vs BBB) don't collide
        off = (8, 6) if i % 2 else (-12, -30)
        a2.annotate(f"{lab}\n{s:.0f} bp -> {p:.2f}", (s, p), textcoords="offset points", xytext=off,
                    fontsize=8, color=INK_2, ha="left" if i % 2 else "right")
    a2.set_xlabel("Z-spread over Treasuries (bp)")
    a2.set_ylabel("Price per 100 face")
    a2.set_title(f"{bond_label}: wider spread = lower price")
    fig.tight_layout()
    return _save(fig, "credit_spread_price.png")
