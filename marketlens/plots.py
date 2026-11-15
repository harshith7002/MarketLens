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
