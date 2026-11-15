"""Phase 3 - options pricing and hedging.

Contents
--------
bs_price / bs_greeks ...... Black-Scholes(-Merton) European call/put + Greeks
mc_price .................. Monte Carlo under Geometric Brownian Motion
mc_convergence ............ error of MC vs Black-Scholes as paths grow
implied_vol ............... Newton-Raphson with bisection fallback
delta_hedge ............... short a call, delta-hedge it, measure the P&L

Notation (used everywhere)
--------------------------
S  spot price          K  strike           T  time to expiry in YEARS
r  risk-free rate (continuous, annual)      q  dividend yield (continuous)
sigma  volatility (annual)                  kind  "call" or "put"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ===========================================================================
# 1. Black-Scholes
# ===========================================================================
def _d1_d2(S, K, T, r, sigma, q=0.0):
    """d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T).

    Intuition: d2 is "how many standard deviations the forward price is
    above the strike". N(d2) is the risk-neutral probability the option
    finishes in the money; N(d1) is the call's delta (with q = 0).
    """
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, kind: str = "call", q: float = 0.0):
    """European option price.

    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    Read the call as: (value of receiving the stock if exercised)
                    - (value of paying the strike if exercised).
    Works element-wise on numpy arrays. At expiry (T <= 0) returns payoff.
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    if np.all(T <= 0):
        return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if kind == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, kind: str = "call", q: float = 0.0) -> dict:
    """All five Greeks, in the units traders quote them.

    delta  dV/dS              change in price per 1 unit move in the stock
    gamma  d2V/dS2            change in DELTA per 1 unit move in the stock
    vega   dV/dsigma / 100    change in price per 1 vol POINT (20% -> 21%)
    theta  dV/dt / 365        change in price per CALENDAR DAY that passes
    rho    dV/dr / 100        change in price per 1% (100 bp) rise in rates
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    eqt, ert = np.exp(-q * T), np.exp(-r * T)
    sqrt_t = np.sqrt(T)

    gamma = eqt * pdf_d1 / (S * sigma * sqrt_t)          # same for call & put
    vega = S * eqt * pdf_d1 * sqrt_t                     # same for call & put
    decay = -S * eqt * pdf_d1 * sigma / (2 * sqrt_t)     # time decay of optionality

    if kind == "call":
        delta = eqt * norm.cdf(d1)
        theta = decay - r * K * ert * norm.cdf(d2) + q * S * eqt * norm.cdf(d1)
        rho = K * T * ert * norm.cdf(d2)
    elif kind == "put":
        delta = -eqt * norm.cdf(-d1)
        theta = decay + r * K * ert * norm.cdf(-d2) - q * S * eqt * norm.cdf(-d1)
        rho = -K * T * ert * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return {
        "price": bs_price(S, K, T, r, sigma, kind, q),
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,
        "theta": theta / 365,
        "rho": rho / 100,
    }


# ===========================================================================
# 2. Monte Carlo under Geometric Brownian Motion
# ===========================================================================
def gbm_terminal(S, T, r, sigma, n_paths, rng, q=0.0, antithetic=False):
    """Simulate S_T under the risk-neutral measure.

    GBM: dS = (r - q) S dt + sigma S dW. Its exact solution is
        S_T = S exp[(r - q - sigma^2/2) T + sigma sqrt(T) Z],  Z ~ N(0,1)
    so for a European option we can jump straight to expiry - no time steps.
    The -sigma^2/2 term is the "volatility drag" that makes E[S_T] = S e^{(r-q)T}.
    """
    if antithetic:  # pair every Z with -Z: same distribution, lower variance
        z = rng.standard_normal(n_paths // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_price(S, K, T, r, sigma, kind="call", n_paths=100_000, seed=42, q=0.0, antithetic=False):
    """Monte Carlo price = e^{-rT} * average payoff. Returns (price, standard error).

    Standard error = std(discounted payoffs) / sqrt(n): to halve the error you
    need FOUR times the paths - the famous 1/sqrt(N) convergence.
    """
    rng = np.random.default_rng(seed)
    ST = gbm_terminal(S, T, r, sigma, n_paths, rng, q, antithetic)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T) * payoff
    if antithetic:  # pairs are dependent: compute the SE on pair averages
        half = len(disc) // 2
        pairs = 0.5 * (disc[:half] + disc[half:])
        return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(half))
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths))


def mc_convergence(S, K, T, r, sigma, kind="call",
                   path_counts=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000),
                   n_seeds=20):
    """For each path count, run `n_seeds` independent MC pricings and record
    the mean absolute error vs Black-Scholes, the mean standard error, and
    one representative estimate (seed 0)."""
    exact = float(bs_price(S, K, T, r, sigma, kind))
    rows = []
    for n in path_counts:
        ests = [mc_price(S, K, T, r, sigma, kind, n, seed=s) for s in range(n_seeds)]
        prices = np.array([p for p, _ in ests])
        rows.append({
            "paths": n,
            "mc_price_seed0": prices[0],
            "bs_price": exact,
            "mean_abs_error": float(np.mean(np.abs(prices - exact))),
            "mean_abs_error_pct": float(np.mean(np.abs(prices - exact)) / exact),
            "mean_std_error": float(np.mean([se for _, se in ests])),
        })
    return rows


# ===========================================================================
# 3. Implied volatility
# ===========================================================================
@dataclass
class IVResult:
    sigma: float          # NaN if the price is outside no-arbitrage bounds
    iterations: int
    method: str           # "newton", "bisection", or "no-arbitrage violation"


def implied_vol(price, S, K, T, r, kind="call", q=0.0,
                tol=1e-10, max_iter=100, lo=1e-4, hi=5.0,
                min_time_value=1e-6) -> IVResult:
    """Find sigma such that bs_price(sigma) == market price.

    There is no closed form, so we solve f(sigma) = BS(sigma) - price = 0.

    Newton-Raphson:  sigma_{n+1} = sigma_n - f(sigma_n) / vega(sigma_n)
        Very fast (quadratic convergence) because vega is the exact slope.
        Fails when vega ~ 0 (deep in/out of the money, or near expiry): the
        step explodes.
    Bisection fallback: price is strictly increasing in sigma, so if
        BS(lo) < price < BS(hi) the root is in [lo, hi]; halve the interval
        until it is tiny. Slow but cannot fail.
    """
    # No-arbitrage bounds: a call is worth at least its discounted intrinsic
    # value and at most the (discounted) stock. Outside these, no sigma exists.
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(disc_S - disc_K, 0.0) if kind == "call" else max(disc_K - disc_S, 0.0)
    upper = disc_S if kind == "call" else disc_K
    if not (lower < price < upper):
        return IVResult(np.nan, 0, "no-arbitrage violation")
    # Time value = price - intrinsic. If it is (numerically) zero, EVERY small
    # sigma reproduces the price, so the implied vol is not identifiable.
    # (e.g. a deep OTM option worth 1e-16, or deep ITM worth only intrinsic.)
    if price - lower < min_time_value * S:
        return IVResult(np.nan, 0, "no time value")

    # --- Newton-Raphson ------------------------------------------------------
    # Starting point: the larger of
    #  * Brenner-Subrahmanyam  sigma ~ sqrt(2 pi / T) * price / S  (good at the money)
    #  * Manaster-Koehler      sigma ~ sqrt(2 |ln(F/K)| / T)       (the inflection
    #    point of price vs sigma; Newton started there converges for any
    #    strike - important for out-of-the-money wings where vega is small)
    F = S * np.exp((r - q) * T)
    sigma = np.clip(max(np.sqrt(2 * np.pi / T) * price / S,
                        np.sqrt(2 * abs(np.log(F / K)) / T)), 0.05, 3.0)
    for i in range(1, max_iter + 1):
        diff = float(bs_price(S, K, T, r, sigma, kind, q)) - price
        if abs(diff) < tol:
            return IVResult(float(sigma), i, "newton")
        vega = float(bs_greeks(S, K, T, r, sigma, kind, q)["vega"]) * 100
        if vega < 1e-8:
            break                                  # flat slope -> Newton unsafe
        sigma_new = sigma - diff / vega
        if not (lo < sigma_new < hi):
            break                                  # step left the sane range
        sigma = sigma_new

    # --- Bisection fallback ------------------------------------------------
    a, b = lo, hi
    for j in range(1, 200):
        mid = 0.5 * (a + b)
        diff = float(bs_price(S, K, T, r, mid, kind, q)) - price
        if abs(diff) < tol or (b - a) < 1e-10:
            return IVResult(mid, j, "bisection")
        if diff > 0:
            b = mid
        else:
            a = mid
    return IVResult(0.5 * (a + b), j, "bisection")


def implied_forward(chain, T: float, r: float, n_nearest: int = 6) -> dict:
    """Implied forward and dividend yield from REAL prices via put-call parity.

    Parity:  C - P = e^{-rT} (F - K)   =>   F = K + e^{rT} (C - P)
    F = S e^{(r-q)T} is the forward price. Using the strikes nearest the money
    (most liquid, and where American early exercise matters least), each
    strike gives an estimate of F; we average them, then q = r - ln(F/S)/T.
    This is how desks get the dividend yield the market is pricing in.

    Diagnostic: the slope of (C - P) vs K across a wider range of strikes
    should be -e^{-rT} (~ -1). For American options it is steeper, because
    deep in-the-money puts are worth more than European ones (early exercise).
    """
    S = float(chain["spot"].iloc[0])
    liquid = chain[chain["bid"] > 0]
    calls = liquid[liquid["type"] == "call"].set_index("strike")["mid"]
    puts = liquid[liquid["type"] == "put"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    near = both[np.argsort(np.abs(both.to_numpy() - S))[:n_nearest]]
    F_each = near.to_numpy(dtype=float) + np.exp(r * T) * (calls[near] - puts[near]).to_numpy()
    F = float(F_each.mean())

    wide = both[(both > S * 0.95) & (both < S * 1.05)]
    slope = float(np.polyfit(wide.to_numpy(dtype=float), (calls[wide] - puts[wide]).to_numpy(), 1)[0])
    return {"spot": S, "forward": F, "q": float(r - np.log(F / S) / T),
            "forward_std": float(F_each.std(ddof=1)), "n_strikes": int(len(near)),
            "parity_slope": slope, "parity_slope_theory": float(-np.exp(-r * T))}


def iv_smile(chain, T: float, r: float, q: float, band: float = 0.20, min_oi: int = 10):
    """Implied vol for every liquid out-of-the-money option.

    OTM options are used because they are the liquid ones: puts below the
    forward, calls above it. Returns a DataFrame sorted by strike.
    """
    import pandas as pd

    S = float(chain["spot"].iloc[0])
    F = S * np.exp((r - q) * T)
    df = chain[(chain["bid"] > 0) & (chain["openInterest"] >= min_oi)].copy()
    df = df[(df["strike"] > S * (1 - band)) & (df["strike"] < S * (1 + band))]
    df = df[((df["type"] == "put") & (df["strike"] < F)) | ((df["type"] == "call") & (df["strike"] >= F))]
    res = [implied_vol(row.mid, S, row.strike, T, r, row.type, q) for row in df.itertuples()]
    df["iv"] = [x.sigma for x in res]
    df["iv_method"] = [x.method for x in res]
    df["iv_iters"] = [x.iterations for x in res]
    df["moneyness"] = df["strike"] / F
    return pd.DataFrame(df).dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)


# ===========================================================================
# 4. Delta hedging
# ===========================================================================
def gbm_paths(S0, T, r, sigma, n_steps, n_paths, rng, mu=None):
    """Full GBM paths, shape (n_steps + 1, n_paths). `mu` = real-world drift
    (defaults to r). Hedging P&L should NOT depend on mu - a key result."""
    mu = r if mu is None else mu
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = S0
    paths[1:] = S0 * np.exp(np.cumsum(log_inc, axis=0))
    return paths


def delta_hedge(paths, K, T, r, sigma_hedge, cost_bps: float = 0.0, q: float = 0.0) -> dict:
    """Sell one European call at t=0 and delta-hedge it at every step of `paths`.

    paths : (n_steps + 1, n_paths) array of stock prices (or a 1-D single path)

    Book-keeping (all per 1 option):
      t=0   receive premium C0, buy delta_0 shares, cash = C0 - delta_0 S_0
      each step: cash grows at r; rebalance to the new delta, paying
                 (new - old delta) * S, plus a transaction cost in bps
      T     hand over the payoff max(S_T - K, 0), sell the shares
      P&L = cash + delta * S_T - payoff

    If the stock really moves with volatility sigma_hedge and we hedge
    continuously with no costs, P&L = 0: the option is perfectly replicated.
    That is the whole argument behind Black-Scholes.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[:, None]
    n_steps = paths.shape[0] - 1
    dt = T / n_steps
    times_left = T - dt * np.arange(n_steps)          # time to expiry at each rebalance
    cost = cost_bps / 1e4

    S0 = paths[0]
    premium = bs_price(S0, K, T, r, sigma_hedge, "call", q)
    delta = bs_greeks(S0, K, T, r, sigma_hedge, "call", q)["delta"]
    cash = premium - delta * S0 - cost * np.abs(delta) * S0
    deltas = [delta]
    costs = cost * np.abs(delta) * S0

    for i in range(1, n_steps):
        S = paths[i]
        cash = cash * np.exp(r * dt) + q * delta * S * dt   # interest (+ dividends on shares)
        new_delta = bs_greeks(S, K, times_left[i], r, sigma_hedge, "call", q)["delta"]
        trade = new_delta - delta
        cash -= trade * S + cost * np.abs(trade) * S
        costs = costs + cost * np.abs(trade) * S
        delta = new_delta
        deltas.append(delta)

    ST = paths[-1]
    cash = cash * np.exp(r * dt) + q * delta * ST * dt
    payoff = np.maximum(ST - K, 0.0)
    # unwind the hedge (sell remaining shares, paying cost) and pay the payoff
    pnl = cash + delta * ST - cost * np.abs(delta) * ST - payoff
    return {
        "pnl": pnl.squeeze(),
        "premium": premium.squeeze(),
        "deltas": np.array(deltas).squeeze(),
        "payoff": payoff.squeeze(),
        "costs": (costs + cost * np.abs(delta) * ST).squeeze(),
    }


def delta_hedge_path_detail(path, K, T, r, sigma_hedge, q=0.0):
    """Day-by-day breakdown for ONE path: mark-to-market P&L of the hedged book.

    Each day the book is: cash + delta * S - (value of the short call at sigma_hedge).
    Its change is the daily hedging P&L; it should hover near zero and drift
    according to realised vs implied volatility (the gamma-theta trade-off).
    """
    path = np.asarray(path, dtype=float)
    n = len(path) - 1
    dt = T / n
    cash, delta, rows = 0.0, 0.0, []
    for i in range(n + 1):
        S, tau = path[i], T - i * dt
        if i > 0:
            cash = cash * np.exp(r * dt) + q * delta * S * dt
        if i == 0:
            cash = float(bs_price(S, K, T, r, sigma_hedge, "call", q))
        if i < n:
            g = bs_greeks(S, K, tau, r, sigma_hedge, "call", q)
            new_delta, opt_val, gamma = float(g["delta"]), float(g["price"]), float(g["gamma"])
            cash -= (new_delta - delta) * S
            delta = new_delta
        else:
            opt_val, gamma = max(S - K, 0.0), 0.0
        rows.append({"step": i, "S": S, "delta": delta, "gamma": gamma,
                     "option_value": opt_val, "book_pnl": cash + delta * S - opt_val})
    return rows
