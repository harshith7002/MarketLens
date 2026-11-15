"""Phase 1 - market data: download, clean, compute log returns, cache to CSV.

Pipeline
--------
1. download_prices()  -> raw adjusted closes from yfinance (one column per asset)
2. clean_prices()     -> align to the NSE calendar, fill short gaps, drop the rest
3. log_returns()      -> r_t = ln(P_t / P_{t-1})
4. load_data()        -> the function every other module calls; reads the CSV
                         cache so the project works offline.

Run `python -m marketlens.data` to (re)download and rebuild the cache.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from marketlens import config


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_prices(
    tickers: dict[str, str] = config.TICKERS,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance.

    `auto_adjust=True` makes yfinance back-adjust the Close column for
    dividends and splits/bonus issues. Without this, a 1:1 bonus issue would
    look like a -50% crash and wreck every risk number downstream.
    """
    import yfinance as yf  # imported here so offline use never needs it

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data - check your connection.")

    prices = raw["Close"].rename(columns=tickers)   # '^NSEI' -> 'NIFTY50', ...
    prices = prices[list(tickers.values())]         # stable column order
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    missing = set(tickers.values()) - set(prices.columns[prices.notna().any()])
    if missing:
        raise RuntimeError(f"No data downloaded for: {sorted(missing)}")
    return prices


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_prices(
    raw: pd.DataFrame,
    anchor: str = config.CALENDAR_ANCHOR,
    ffill_limit: int = config.FFILL_LIMIT,
) -> tuple[pd.DataFrame, dict]:
    """Put every asset on one common calendar and remove gaps.

    Why gaps exist: the assets trade on different calendars.
      * NSE stocks/index close on Indian holidays (Diwali, Holi, ...).
      * Gold futures (COMEX) close on US holidays (Thanksgiving, July 4th).
      * USD/INR quotes even on some days the NSE is shut, and on odd weekends.

    Rules (simple and defensible):
      a) Keep only days the anchor (NIFTY 50) traded -> one NSE calendar.
      b) If FX/gold is missing on an NSE day (e.g. US holiday), carry the last
         known price forward, at most `ffill_limit` days. "Price unchanged
         that day" is a reasonable assumption for a 1-day gap; it is NOT for a
         long gap, so longer gaps stay NaN.
      c) Drop any row that still has a NaN, so every row is a complete
         cross-section (needed for correlations and portfolio VaR).
    """
    report: dict = {"raw_rows": len(raw), "raw_missing": raw.isna().sum().to_dict()}

    # a) align to the NSE calendar
    px = raw[raw[anchor].notna()].copy()
    report["rows_on_anchor_calendar"] = len(px)
    report["missing_on_anchor_calendar"] = px.isna().sum().to_dict()

    # b) forward-fill short gaps only
    px = px.ffill(limit=ffill_limit)

    # c) drop incomplete rows
    before = len(px)
    px = px.dropna(how="any")
    report["rows_dropped_incomplete"] = before - len(px)

    # Sanity: prices must be strictly positive (log of <=0 is undefined)
    if (px <= 0).any().any():
        bad = px.columns[(px <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found in {bad}")

    report["clean_rows"] = len(px)
    report["first_date"] = px.index.min().strftime("%Y-%m-%d")
    report["last_date"] = px.index.max().strftime("%Y-%m-%d")
    return px, report


# ---------------------------------------------------------------------------
# 3. Returns
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns r_t = ln(P_t / P_{t-1}).

    The first row has no previous price, so it is dropped.
    """
    return np.log(prices / prices.shift(1)).dropna(how="any")


def flag_outliers(returns: pd.DataFrame, threshold: float = config.OUTLIER_LOG_RET) -> pd.DataFrame:
    """List days where |log return| > threshold, for a human to check.

    We do NOT delete them automatically: big moves (elections, crashes) are
    real and are exactly what risk models must see. We only flag them in case
    one is a data error (e.g. an unadjusted split).
    """
    stacked = returns.stack()
    big = stacked[stacked.abs() > threshold]
    return big.rename("log_return").reset_index().rename(columns={"level_1": "Asset"})


# ---------------------------------------------------------------------------
# 4. Cache + load
# ---------------------------------------------------------------------------
def build_cache() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Download, clean, compute returns and write all three files to data/."""
    raw = download_prices()
    prices, report = clean_prices(raw)
    rets = log_returns(prices)

    report["return_rows"] = len(rets)
    report["outliers"] = flag_outliers(rets).astype(str).to_dict(orient="records")
    report["source"] = "Yahoo Finance via yfinance (auto_adjust=True)"
    report["tickers"] = config.TICKERS
    report["requested_range"] = [config.START_DATE, config.END_DATE]
    report["downloaded_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    config.DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(config.PRICES_CSV, float_format="%.6f")
    rets.to_csv(config.RETURNS_CSV, float_format="%.8f")
    config.META_JSON.write_text(json.dumps(report, indent=2, default=str))
    return prices, rets, report


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, log_returns). Uses the CSV cache unless refresh=True.

    This is the only entry point the rest of the project uses, so everything
    downstream runs offline once data/ exists.
    """
    if refresh or not (config.PRICES_CSV.exists() and config.RETURNS_CSV.exists()):
        prices, rets, _ = build_cache()
        return prices, rets
    prices = pd.read_csv(config.PRICES_CSV, index_col="Date", parse_dates=True)
    rets = pd.read_csv(config.RETURNS_CSV, index_col="Date", parse_dates=True)
    return prices, rets


# ---------------------------------------------------------------------------
# 5. Option chain snapshot (for the implied-volatility smile)
# ---------------------------------------------------------------------------
def download_option_chain(ticker: str = config.OPTION_TICKER,
                          target_days: int = config.OPTION_TARGET_DAYS) -> pd.DataFrame:
    """Snapshot one expiry of a real option chain and cache it to CSV.

    Yahoo has no NIFTY option chain, so we use SPY (S&P 500 ETF), the most
    liquid options market in the world. It is a one-off snapshot: the IV
    smile in the report is reproducible from the cached file.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    today = pd.Timestamp.now().normalize()
    expiries = pd.to_datetime(pd.Series(tk.options))
    expiry = expiries.iloc[(expiries - today - pd.Timedelta(days=target_days)).abs().argmin()]
    chain = tk.option_chain(expiry.strftime("%Y-%m-%d"))
    hist = tk.history(period="5d")["Close"]
    spot = float(hist.iloc[-1])
    quote_date = hist.index[-1].strftime("%Y-%m-%d")  # last close the quotes belong to

    frames = []
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
        df = df[["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]].copy()
        df["type"] = kind
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["expiry"] = expiry.strftime("%Y-%m-%d")
    out["spot"] = spot
    out["quote_date"] = quote_date
    out["fetched_at_utc"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    config.DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OPTIONS_CSV, index=False)
    return out


def load_option_chain(refresh: bool = False) -> pd.DataFrame:
    if refresh or not config.OPTIONS_CSV.exists():
        return download_option_chain()
    return pd.read_csv(config.OPTIONS_CSV)


if __name__ == "__main__":
    prices, rets, rep = build_cache()
    print(f"Saved {config.PRICES_CSV.name}: {prices.shape}, {config.RETURNS_CSV.name}: {rets.shape}")
    print(f"Range: {rep['first_date']} -> {rep['last_date']}")
    print("Missing values on NSE calendar before cleaning:", rep["missing_on_anchor_calendar"])
    print("Rows dropped as incomplete:", rep["rows_dropped_incomplete"])
    print("Flagged moves (|log ret| > 15%):", rep["outliers"] or "none")
    print("NaNs in returns:", int(rets.isna().sum().sum()))
