"""Phase 2 - risk analytics for equities and FX.

Contents
--------
Per-asset metrics ........ annualised return, volatility, Sharpe, max drawdown
Dependence ............... correlation matrix, rolling correlation
Portfolio VaR / ES ....... historical, parametric (variance-covariance), Monte Carlo
Backtesting .............. rolling out-of-sample VaR, breach count, Kupiec POF test

Conventions
-----------
* `returns` are daily LOG returns (from data.py).
* VaR and ES are reported as POSITIVE numbers = a loss, as a fraction of
  portfolio value (multiply by the notional to get rupees).
* Horizon is 1 trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from marketlens import config

TD = config.TRADING_DAYS


# ===========================================================================
# 1. Per-asset performance & risk metrics
# ===========================================================================
def annualised_return(log_rets: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Compound annual growth rate (CAGR).

    Log returns add over time, so the average yearly log return is
    252 * mean(daily log return). exp(...) - 1 turns it back into a normal %.
    """
    return np.exp(log_rets.mean() * TD) - 1


def annualised_volatility(log_rets: pd.DataFrame | pd.Series) -> pd.Series | float:
    """sigma_annual = sigma_daily * sqrt(252).

    Variance grows linearly with time if daily returns are independent
    (Var of a sum of 252 iid terms = 252 * Var), so std grows with sqrt(time).
    """
    return log_rets.std() * np.sqrt(TD)


def sharpe_ratio(log_rets: pd.DataFrame | pd.Series, rf: float = config.RISK_FREE_RATE):
    """Sharpe = (annual return - risk-free) / annual volatility.

    "How much extra return did I earn per unit of risk taken?" Both returns
    are in continuous (log) terms so they are directly comparable.
    """
    excess = log_rets.mean() * TD - np.log1p(rf)
    return excess / annualised_volatility(log_rets)


def max_drawdown(prices: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Worst peak-to-trough fall: min over t of P_t / max(P_0..P_t) - 1."""
    return (prices / prices.cummax() - 1).min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1


def asset_summary(prices: pd.DataFrame, log_rets: pd.DataFrame) -> pd.DataFrame:
    """One row per asset with all four headline metrics."""
    return pd.DataFrame(
        {
            "Ann. return": annualised_return(log_rets),
            "Ann. volatility": annualised_volatility(log_rets),
            "Sharpe": sharpe_ratio(log_rets),
            "Max drawdown": max_drawdown(prices),
        }
    )


# ===========================================================================
# 2. Correlation
# ===========================================================================
def correlation_matrix(log_rets: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily log returns (-1 .. +1)."""
    return log_rets.corr()


def rolling_correlation(
    log_rets: pd.DataFrame, a: str, b: str, window: int = config.ROLLING_CORR_WINDOW
) -> pd.Series:
    """Correlation of a and b over a trailing `window`-day window.

    Shows that correlation is NOT constant - it jumps in stress periods,
    which is exactly when diversification is needed most.
    """
    return log_rets[a].rolling(window).corr(log_rets[b]).dropna()


# ===========================================================================
# 3. Portfolio returns
# ===========================================================================
def portfolio_returns(log_rets: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily SIMPLE return of a portfolio rebalanced to fixed weights.

    Log returns do not add across assets, simple returns do:
        R_p = sum_i w_i * R_i,   R_i = exp(r_i) - 1
    """
    w = pd.Series(weights)
    simple = np.expm1(log_rets[w.index])
    return simple @ w


# ===========================================================================
# 4. Value-at-Risk & Expected Shortfall
# ===========================================================================
# VaR_a  = the loss that is exceeded only (1-a) of the time.
#          "On 99 days out of 100 we lose less than VaR_99."
# ES_a   = the AVERAGE loss on the days that are worse than VaR_a.
#          It answers "when things go wrong, how bad on average?"
# ---------------------------------------------------------------------------
def historical_var_es(port_rets: pd.Series | np.ndarray, level: float) -> tuple[float, float]:
    """Historical simulation: just read the quantile off past returns.

    No distribution assumed - fat tails and skew are captured automatically,
    but the answer can only be as bad as the worst day in the window.
    """
    r = np.asarray(port_rets)
    q = np.quantile(r, 1 - level)          # e.g. the 1st percentile for 99%
    var = -q
    es = -r[r <= q].mean()
    return float(var), float(es)


def parametric_var_es(
    log_rets: pd.DataFrame, weights: dict[str, float], level: float
) -> tuple[float, float]:
    """Variance-covariance (delta-normal) method.

    Assume asset returns are jointly normal with mean vector mu and
    covariance matrix Sigma. Then the portfolio return is normal with
        mu_p    = w' mu
        sigma_p = sqrt(w' Sigma w)          <- correlation enters here
    and
        VaR = z_a * sigma_p - mu_p          (z_99 = 2.326, z_95 = 1.645)
        ES  = sigma_p * phi(z_a) / (1 - a) - mu_p
    """
    w = pd.Series(weights)
    simple = np.expm1(log_rets[w.index])
    mu_p = float(simple.mean() @ w)
    sigma_p = float(np.sqrt(w @ simple.cov() @ w))
    z = stats.norm.ppf(level)
    var = z * sigma_p - mu_p
    es = sigma_p * stats.norm.pdf(z) / (1 - level) - mu_p
    return var, es


def monte_carlo_var_es(
    log_rets: pd.DataFrame,
    weights: dict[str, float],
    level: float,
    n_sims: int = config.MC_SIMS,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Monte Carlo simulation of tomorrow's portfolio return.

    1. Estimate mean vector mu and covariance Sigma of the log returns.
    2. Cholesky: Sigma = L L'. If Z ~ N(0, I) then mu + L Z has covariance
       Sigma - this is how we create CORRELATED random draws.
    3. Turn each simulated log-return vector into a portfolio P&L
       (exp(r) - 1, weighted), then take the quantile as in historical VaR.

    With a normal model this should land close to parametric VaR; its power
    is that step 3 can revalue ANY portfolio, including options (non-linear).
    """
    rng = rng or np.random.default_rng(config.SEED)
    w = pd.Series(weights)
    x = log_rets[w.index]
    mu = x.mean().to_numpy()
    L = np.linalg.cholesky(x.cov().to_numpy())
    z = rng.standard_normal((n_sims, len(w)))
    sim_log = mu + z @ L.T
    sim_port = np.expm1(sim_log) @ w.to_numpy()
    return historical_var_es(sim_port, level)


def var_table(
    log_rets: pd.DataFrame,
    weights: dict[str, float] = config.PORTFOLIO_WEIGHTS,
    levels=config.VAR_LEVELS,
    window: int = config.VAR_WINDOW,
    notional: float = config.PORTFOLIO_VALUE_INR,
) -> pd.DataFrame:
    """Current 1-day VaR and ES (all 3 methods, both levels) on the last `window` days."""
    recent = log_rets.iloc[-window:]
    port = portfolio_returns(recent, weights)
    rows = []
    for lvl in levels:
        for method, (v, e) in {
            "Historical": historical_var_es(port, lvl),
            "Parametric": parametric_var_es(recent, weights, lvl),
            "Monte Carlo": monte_carlo_var_es(recent, weights, lvl),
        }.items():
            rows.append(
                {"Method": method, "Level": f"{lvl:.0%}", "VaR %": v, "ES %": e,
                 "VaR INR": v * notional, "ES INR": e * notional}
            )
    return pd.DataFrame(rows)


# ===========================================================================
# 5. Backtesting
# ===========================================================================
def rolling_var_forecasts(
    log_rets: pd.DataFrame,
    weights: dict[str, float] = config.PORTFOLIO_WEIGHTS,
    levels=config.VAR_LEVELS,
    window: int = config.VAR_WINDOW,
    test_days: int = config.BACKTEST_DAYS,
    n_sims: int = config.MC_SIMS,
) -> pd.DataFrame:
    """Out-of-sample VaR forecasts for each of the last `test_days` days.

    For day t we use ONLY returns from t-window .. t-1 (no look-ahead),
    forecast VaR, then compare with the loss that actually happened on day t.
    """
    rng = np.random.default_rng(config.SEED)
    port = portfolio_returns(log_rets, weights)
    n = len(log_rets)
    if n < window + test_days:
        raise ValueError("Not enough history for this window/backtest length.")

    out = []
    for t in range(n - test_days, n):
        hist = log_rets.iloc[t - window : t]        # strictly before day t
        hist_port = port.iloc[t - window : t]
        row = {"Date": log_rets.index[t], "Realised": port.iloc[t]}
        for lvl in levels:
            tag = f"{int(lvl * 100)}"
            row[f"Historical_{tag}"] = historical_var_es(hist_port, lvl)[0]
            row[f"Parametric_{tag}"] = parametric_var_es(hist, weights, lvl)[0]
            row[f"Monte Carlo_{tag}"] = monte_carlo_var_es(hist, weights, lvl, n_sims, rng)[0]
        out.append(row)
    return pd.DataFrame(out).set_index("Date")


def kupiec_pof(breaches: int, n_obs: int, level: float) -> tuple[float, float]:
    """Kupiec Proportion-Of-Failures likelihood-ratio test.

    H0: the true breach probability is p = 1 - level (model is calibrated).
    With x breaches in T days:
        LR = -2 ln[ (1-p)^(T-x) p^x ] + 2 ln[ (1-x/T)^(T-x) (x/T)^x ]
    Under H0, LR ~ chi-squared with 1 degree of freedom.
    p-value < 0.05 -> reject: the model has too many OR too few breaches.
    """
    p = 1 - level
    x, T = breaches, n_obs
    phat = x / T

    def _ll(prob: float) -> float:  # log-likelihood, with 0*log(0) := 0
        a = (T - x) * np.log1p(-prob) if T - x > 0 else 0.0
        b = x * np.log(prob) if x > 0 else 0.0
        return a + b

    lr = -2 * (_ll(p) - _ll(phat))
    lr = max(lr, 0.0)  # guard tiny negative from rounding
    return float(lr), float(stats.chi2.sf(lr, df=1))


def basel_zone(breaches: int, n_obs: int = 250) -> str:
    """Basel traffic light for a 99% VaR over 250 days (green 0-4, yellow 5-9, red 10+)."""
    if n_obs != 250:
        return "n/a"
    return "Green" if breaches <= 4 else "Yellow" if breaches <= 9 else "Red"


def backtest_summary(forecasts: pd.DataFrame, levels=config.VAR_LEVELS) -> pd.DataFrame:
    """Breach counts and Kupiec results for every method/level."""
    T = len(forecasts)
    loss = -forecasts["Realised"]
    rows = []
    for lvl in levels:
        tag = f"{int(lvl * 100)}"
        for method in ("Historical", "Parametric", "Monte Carlo"):
            x = int((loss > forecasts[f"{method}_{tag}"]).sum())
            lr, pval = kupiec_pof(x, T, lvl)
            rows.append(
                {"Method": method, "Level": f"{lvl:.0%}", "Days": T,
                 "Expected": (1 - lvl) * T, "Breaches": x,
                 "Kupiec LR": lr, "p-value": pval,
                 "Result": "Calibrated" if pval >= 0.05 else "Rejected",
                 "Basel zone": basel_zone(x, T) if lvl == 0.99 else "-"}
            )
    return pd.DataFrame(rows)
