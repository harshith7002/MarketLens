"""Central configuration for MarketLens.

Every module imports its settings from here, so changing a ticker or a date
range happens in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PRICES_CSV = DATA_DIR / "prices.csv"
RETURNS_CSV = DATA_DIR / "returns.csv"
META_JSON = DATA_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Universe: yfinance ticker -> short name used everywhere else in the project
# ---------------------------------------------------------------------------
TICKERS = {
    "^NSEI": "NIFTY50",        # NIFTY 50 index (India's large-cap benchmark)
    "RELIANCE.NS": "RELIANCE",  # Energy / telecom / retail conglomerate
    "TCS.NS": "TCS",            # IT services
    "HDFCBANK.NS": "HDFCBANK",  # Private-sector bank
    "INFY.NS": "INFY",          # IT services
    "ICICIBANK.NS": "ICICIBANK",  # Private-sector bank
    "INR=X": "USDINR",          # Rupees per 1 US dollar (up = rupee weakens)
    "GC=F": "GOLD",             # COMEX gold front-month future, USD per troy oz
}

# The asset whose trading calendar we align everything to. The project is
# India-focused, so we keep the days the NSE was open.
CALENDAR_ANCHOR = "NIFTY50"

# The five single stocks (used later for the equity portfolio).
EQUITIES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

# ---------------------------------------------------------------------------
# Date range. Fixed (not "today") so every number in the README is
# reproducible from the cached CSVs. yfinance treats `end` as exclusive.
# ---------------------------------------------------------------------------
START_DATE = "2021-11-15"
END_DATE = "2026-11-14"  # last bar = Fri 2026-11-13

# Cleaning rules
FFILL_LIMIT = 3          # max consecutive days we carry a stale price forward
OUTLIER_LOG_RET = 0.15   # |daily log return| above this is flagged for review

TRADING_DAYS = 252       # used to annualise daily statistics

# ---------------------------------------------------------------------------
# Phase 2 - risk settings
# ---------------------------------------------------------------------------
# ASSUMPTION: flat annual risk-free rate (roughly an Indian 91-day T-bill
# yield). Not downloaded - change it here and every Sharpe ratio updates.
RISK_FREE_RATE = 0.06

# Portfolio for VaR: equal-weight the five NSE stocks, Rs 1 crore notional.
PORTFOLIO_WEIGHTS = {name: 1 / len(EQUITIES) for name in EQUITIES}
PORTFOLIO_VALUE_INR = 10_000_000

VAR_LEVELS = (0.95, 0.99)
VAR_WINDOW = 250          # look-back days used to estimate each VaR (~1 year)
BACKTEST_DAYS = 250       # out-of-sample days in the backtest (~last year)
MC_SIMS = 10_000
SEED = 42
ROLLING_CORR_WINDOW = 60

FIG_DIR = ROOT_DIR / "reports" / "figures"
