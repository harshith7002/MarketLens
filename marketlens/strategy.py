"""Phase 4 - pairs trading (statistical arbitrage) on two related stocks.

Idea
----
Two banks are exposed to the same economy, rates and credit cycle, so their
prices tend to move together. If one runs ahead of the other, bet that the gap
(the "spread") closes: short the expensive one, buy the cheap one. The trade is
roughly market-neutral - if the whole market falls, both legs fall.

Pipeline
--------
engle_granger() ......... is the spread actually mean-reverting? (cointegration)
rolling_spread_z() ...... hedge ratio + z-score using ONLY past data
generate_positions() .... z-score rules -> +1 / 0 / -1 position in the spread
backtest() .............. daily P&L with execution lag and transaction costs
trade_log() / metrics() . total return, Sharpe, drawdown, win rate, ...

No look-ahead
-------------
* Hedge ratio and z-score on day t use prices up to and including t only.
* The signal formed at the close of day t is traded at the close of t+1
  (EXEC_LAG = 1), so it first earns the return of day t+2.
* A test perturbs future prices and checks past positions do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from marketlens import config


# ===========================================================================
# 1. Cointegration test
# ===========================================================================
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger two-step test on LOG prices.

    Step 1  OLS:  ln Y_t = alpha + beta * ln X_t + e_t
    Step 2  ADF test on the residuals e_t.
            H0: e_t has a unit root (a random walk -> the gap can wander off
                forever, no mean reversion).
            p < 0.05 -> reject H0 -> the pair is cointegrated.

    Correlation is about daily RETURNS moving together; cointegration is
    about PRICE LEVELS staying together. Two stocks can be highly correlated
    and still drift apart forever - only cointegration supports pairs trading.

    statsmodels.coint uses Engle-Granger critical values (MacKinnon), which
    are stricter than the ordinary ADF ones because beta was estimated.
    Also returns the half-life of mean reversion from an AR(1) on the residual:
        de_t = phi * e_{t-1} + noise   ->   half-life = -ln 2 / ln(1 + phi)
    """
    ly, lx = np.log(y), np.log(x)
    beta, alpha = np.polyfit(lx, ly, 1)
    resid = ly - alpha - beta * lx
    t_stat, pvalue, crit = coint(ly, lx)
    phi = np.polyfit(resid.shift(1).dropna(), resid.diff().dropna(), 1)[0]
    half_life = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.inf
    return {"beta": float(beta), "alpha": float(alpha), "t_stat": float(t_stat),
            "p_value": float(pvalue), "crit_5pct": float(crit[1]),
            "half_life_days": float(half_life), "n_obs": int(len(ly))}


def pair_screen(prices: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Engle-Granger p-value and return correlation for every pair."""
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            eg = engle_granger(prices[a], prices[b])
            corr = np.log(prices[[a, b]]).diff().corr().iloc[0, 1]
            rows.append({"Y": a, "X": b, "return_corr": corr, "EG_p_value": eg["p_value"],
                         "beta": eg["beta"], "half_life_days": eg["half_life_days"]})
    return pd.DataFrame(rows).sort_values("EG_p_value").reset_index(drop=True)


# ===========================================================================
# 2. Spread and z-score with no look-ahead
# ===========================================================================
def rolling_spread_z(y: pd.Series, x: pd.Series, method: str = "ols",
                     beta_window: int = config.BETA_WINDOW,
                     z_window: int = config.Z_WINDOW) -> pd.DataFrame:
    """Hedge ratio, spread and z-score for each day, using past data only.

    method="ols"   : beta_t = OLS slope of ln Y on ln X over the last
                     `beta_window` days (the Engle-Granger hedge ratio).
    method="ratio" : beta = 1, i.e. the spread is ln(Y / X). Simpler, very
                     stable, and dollar-neutral; a common practitioner baseline.

    z_t = (s_t - mean(s over last z_window)) / std(s over last z_window),
    where the spread history s_k = ln Y_k - beta_t ln X_k is recomputed with
    TODAY's beta, so the z-score is internally consistent.
    """
    ly, lx = np.log(y).to_numpy(), np.log(x).to_numpy()
    n = len(ly)
    start = beta_window if method == "ols" else z_window
    beta = np.full(n, np.nan)
    z = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    for t in range(start - 1, n):
        if method == "ols":
            b = np.polyfit(lx[t - beta_window + 1 : t + 1], ly[t - beta_window + 1 : t + 1], 1)[0]
        else:
            b = 1.0
        s_hist = ly[t - z_window + 1 : t + 1] - b * lx[t - z_window + 1 : t + 1]
        sd = s_hist.std(ddof=1)
        beta[t], spread[t] = b, s_hist[-1]
        z[t] = (s_hist[-1] - s_hist.mean()) / sd if sd > 0 else 0.0
    return pd.DataFrame({"beta": beta, "spread": spread, "z": z}, index=y.index)


# ===========================================================================
# 3. Trading rules
# ===========================================================================
def generate_positions(z: pd.Series, entry: float = config.Z_ENTRY, exit: float = config.Z_EXIT,
                       stop: float = config.Z_STOP) -> pd.Series:
    """Target position in the spread decided at each close.

    +1 = LONG the spread  (buy Y, sell beta of X)  when z < -entry  (Y cheap)
    -1 = SHORT the spread (sell Y, buy beta of X)  when z > +entry  (Y rich)
     0 = flat when |z| < exit (gap closed) or |z| > stop (stop-loss).
    After a stop-loss we wait until |z| < exit before trading again, so we do
    not immediately re-enter a relationship that may have broken.
    """
    pos = np.zeros(len(z))
    cur, locked = 0, False
    for i, zi in enumerate(z.to_numpy()):
        if np.isnan(zi):
            cur = 0
        elif cur != 0:
            if abs(zi) > stop:
                cur, locked = 0, True
            elif abs(zi) < exit or np.sign(zi) == cur:  # converged, or crossed through 0
                cur = 0
        else:
            if locked and abs(zi) < exit:
                locked = False
            if not locked and entry < abs(zi) <= stop:
                cur = -int(np.sign(zi))
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="target_position")


# ===========================================================================
# 4. Backtest
# ===========================================================================
def backtest(y: pd.Series, x: pd.Series, method: str = "ols",
             cost_bps: float = config.COST_BPS, exec_lag: int = config.EXEC_LAG,
             entry: float = config.Z_ENTRY, exit: float = config.Z_EXIT,
             stop: float = config.Z_STOP) -> pd.DataFrame:
    """Daily P&L of the pairs strategy.

    Capital is the GROSS notional. A position of +1 in the spread holds
        w_Y = +1 / (1 + |beta|)     w_X = -beta / (1 + |beta|)
    of capital, so |w_Y| + |w_X| = 1 (Rs 1 of capital = Rs 1 of gross exposure).
    Daily return = w_Y * R_Y + w_X * R_X  (simple returns)
    Cost         = cost_bps * (|change in w_Y| + |change in w_X|)
    The hedge ratio drifts daily, so small rebalancing trades are charged too.
    """
    sz = rolling_spread_z(y, x, method)
    target = generate_positions(sz["z"], entry, exit, stop)
    held = target.shift(exec_lag).fillna(0)          # executed at close t+lag
    beta_used = sz["beta"].shift(exec_lag).ffill().fillna(0)

    w_y = held / (1 + beta_used.abs())
    w_x = -held * beta_used / (1 + beta_used.abs())
    r_y, r_x = y.pct_change().fillna(0), x.pct_change().fillna(0)
    # weights set at close t-1 earn the return from t-1 to t
    gross = w_y.shift(1).fillna(0) * r_y + w_x.shift(1).fillna(0) * r_x
    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    cost = turnover * cost_bps / 1e4
    net = gross - cost
    return pd.DataFrame({
        "Y": y, "X": x, "beta": sz["beta"], "z": sz["z"], "target": target,
        "position": held, "w_y": w_y, "w_x": w_x,
        "gross_ret": gross, "cost": cost, "net_ret": net,
        "equity": (1 + net).cumprod(),
    })


# ===========================================================================
# 5. Performance
# ===========================================================================
def trade_log(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per round-trip trade (from the day the position opens to the day it closes)."""
    pos = bt["position"].to_numpy()
    trades, start = [], None
    for i in range(1, len(pos)):
        if pos[i] != pos[i - 1]:
            if pos[i - 1] != 0 and start is not None:     # a trade just closed
                trades.append((start, i))
                start = None
            if pos[i] != 0:                               # a trade just opened
                start = i
    still_open = start is not None
    if still_open:                                        # still open at the end
        trades.append((start, len(pos) - 1))
    rows = []
    for k, (s, e) in enumerate(trades):
        # returns earned while open: day s+1 .. e (weights set at s), plus costs at s and e
        rets = bt["net_ret"].iloc[s + 1 : e + 1]
        pnl = float((1 + rets).prod() * (1 - bt["cost"].iloc[s]) - 1)
        rows.append({"entry": bt.index[s], "exit": bt.index[e],
                     "side": "long spread" if pos[s] > 0 else "short spread",
                     "days": e - s, "return": pnl,
                     "status": "open" if still_open and k == len(trades) - 1 else "closed"})
    return pd.DataFrame(rows)


def metrics(daily_ret: pd.Series, rf: float = 0.0, trades: pd.DataFrame | None = None) -> dict:
    """Headline statistics from a series of daily simple returns."""
    td = config.TRADING_DAYS
    equity = (1 + daily_ret).cumprod()
    years = len(daily_ret) / td
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    vol = float(daily_ret.std() * np.sqrt(td))
    excess = daily_ret - ((1 + rf) ** (1 / td) - 1)
    sharpe = float(excess.mean() / daily_ret.std() * np.sqrt(td)) if vol > 0 else np.nan
    out = {"Total return": total, "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe,
           "Max drawdown": float((equity / equity.cummax() - 1).min())}
    if trades is not None:
        out.update({
            "Trades": len(trades),
            "Win rate": float((trades["return"] > 0).mean()) if len(trades) else np.nan,
            "Avg trade": float(trades["return"].mean()) if len(trades) else np.nan,
            "Avg holding days": float(trades["days"].mean()) if len(trades) else np.nan,
        })
    return out
