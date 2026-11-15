"""Phase 5 - export MarketLens results to a formatted Excel workbook.

Sheets
------
Summary      headline numbers from every phase
Risk         per-asset return / vol / Sharpe / drawdown + correlation matrix
VaR          VaR & ES table (3 methods x 2 levels) + backtest with Kupiec test
VaR Macro    INPUT cells (weights, window, value) that the VBA macro reads,
             the Python reference numbers, and the area the macro writes to
Greeks       Black-Scholes prices and Greeks across strikes (Python values)
Pricer       live Black-Scholes calculator: native Excel formulas AND the VBA
             user-defined functions side by side (the VBA ones show #NAME?
             until MarketLens.bas is imported - see vba/README_VBA.md)
Pairs        strategy vs buy-and-hold, plus the trade log
Returns      daily simple returns of the 5 stocks (the macro's raw data)

Run:  python -m marketlens.excel_export      -> reports/MarketLens.xlsx
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from marketlens import config, options, risk, strategy
from marketlens.data import load_data

XLSX_PATH = config.ROOT_DIR / "reports" / "MarketLens.xlsx"

# --- styles ---------------------------------------------------------------
NAVY = "1F3A5F"
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
HEADER_FONT = Font(bold=True, color="FFFFFF")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")     # yellow = "you may edit this"
OUTPUT_FILL = PatternFill("solid", fgColor="E2EFDA")    # green  = "the macro writes here"
TITLE_FONT = Font(bold=True, size=14, color=NAVY)
NOTE_FONT = Font(italic=True, size=9, color="595959")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
PCT2, PCT1, NUM2, NUM4, INR = "0.00%", "0.0%", "0.00", "0.0000", '"Rs "#,##0'


def _title(ws, text, note=None):
    ws["A1"] = text
    ws["A1"].font = TITLE_FONT
    if note:
        ws["A2"] = note
        ws["A2"].font = NOTE_FONT


def _table(ws, df: pd.DataFrame, row: int, col: int = 1, fmts: dict | None = None,
           index: bool = True, index_label: str = "") -> tuple[int, int]:
    """Write a DataFrame with a styled header. Returns (last_row, last_col)."""
    fmts = fmts or {}
    headers = ([index_label] if index else []) + [str(c) for c in df.columns]
    for j, h in enumerate(headers):
        c = ws.cell(row=row, column=col + j, value=h)
        c.fill, c.font, c.border = HEADER_FILL, HEADER_FONT, BOX
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, (idx, rec) in enumerate(df.iterrows(), start=1):
        values = ([idx] if index else []) + list(rec.values)
        for j, v in enumerate(values):
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            c = ws.cell(row=row + i, column=col + j, value=v)
            c.border = BOX
            name = headers[j]
            if name in fmts:
                c.number_format = fmts[name]
            elif isinstance(v, float):
                c.number_format = NUM4
            if hasattr(v, "year"):
                c.number_format = "yyyy-mm-dd"
    return row + len(df), col + len(headers) - 1


def _autowidth(ws, min_w=9, max_w=42):
    widths = {}
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None:
                n = len(str(c.value)) if not isinstance(c.value, float) else 10
                widths[c.column_letter] = max(widths.get(c.column_letter, 0), n)
    for col, w in widths.items():
        ws.column_dimensions[col].width = max(min_w, min(max_w, w + 2))


# ===========================================================================
# Sheet builders
# ===========================================================================
def sheet_risk(wb, prices, rets):
    ws = wb.create_sheet("Risk")
    _title(ws, "Per-asset risk & return (5 years, daily data)",
           f"{prices.index[0]:%d %b %Y} - {prices.index[-1]:%d %b %Y}. Sharpe uses rf = "
           f"{config.RISK_FREE_RATE:.0%} (assumption). Return = CAGR from log returns.")
    summ = risk.asset_summary(prices, rets)
    last, _ = _table(ws, summ, 4, fmts={"Ann. return": PCT2, "Ann. volatility": PCT2,
                                         "Sharpe": NUM2, "Max drawdown": PCT2}, index_label="Asset")
    ws.conditional_formatting.add(f"D5:D{last}", ColorScaleRule(
        start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
        mid_color="FFFFFF", end_type="max", end_color="63BE7B"))

    r0 = last + 3
    ws.cell(row=r0 - 1, column=1, value="Correlation of daily log returns").font = Font(bold=True)
    corr = risk.correlation_matrix(rets)
    last, lastc = _table(ws, corr, r0, fmts={c: NUM2 for c in corr.columns})
    rng = f"B{r0 + 1}:{get_column_letter(lastc)}{last}"
    ws.conditional_formatting.add(rng, ColorScaleRule(
        start_type="num", start_value=-1, start_color="E34948", mid_type="num", mid_value=0,
        mid_color="FFFFFF", end_type="num", end_value=1, end_color="2A78D6"))
    ws.freeze_panes = "B5"
    _autowidth(ws)
    return summ


def sheet_var(wb, rets):
    ws = wb.create_sheet("VaR")
    _title(ws, f"Portfolio Value-at-Risk - equal-weight {', '.join(config.EQUITIES)}",
           f"Rs {config.PORTFOLIO_VALUE_INR:,.0f} notional, 1-day horizon, last {config.VAR_WINDOW} days. "
           "VaR/ES are positive numbers = losses.")
    vt = risk.var_table(rets)
    last, _ = _table(ws, vt, 4, index=False, fmts={"VaR %": "0.000%", "ES %": "0.000%",
                                                   "VaR INR": INR, "ES INR": INR})

    # one text label per row for the chart's x-axis, e.g. "Historical 95%"
    ws.cell(row=4, column=8, value="Chart label").font = NOTE_FONT
    for r in range(5, last + 1):
        ws.cell(row=r, column=8, value=f"{ws.cell(row=r, column=1).value} {ws.cell(row=r, column=2).value}").font = NOTE_FONT

    # native Excel bar chart of VaR by method & level
    chart = BarChart()
    chart.type, chart.style = "col", 10
    chart.title, chart.y_axis.title = "1-day VaR and ES (Rs)", "Rs"
    data = Reference(ws, min_col=5, max_col=6, min_row=4, max_row=last)
    cats = Reference(ws, min_col=8, min_row=5, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.x_axis.delete = False          # newer Excel hides openpyxl axes unless told not to
    chart.y_axis.delete = False
    chart.y_axis.numFmt = "#,##0"
    chart.y_axis.majorGridlines = None
    chart.legend.position = "b"
    chart.height, chart.width = 8, 17
    ws.add_chart(chart, "K4")

    r0 = last + 4
    ws.cell(row=r0 - 1, column=1,
            value=f"Backtest: rolling {config.VAR_WINDOW}-day VaR, last {config.BACKTEST_DAYS} days "
                  "(out-of-sample), Kupiec proportion-of-failures test").font = Font(bold=True)
    fc = risk.rolling_var_forecasts(rets)
    bt = risk.backtest_summary(fc)
    last2, _ = _table(ws, bt, r0, index=False, fmts={"Expected": "0.0", "Kupiec LR": "0.000",
                                                     "p-value": "0.000"})
    res_col = get_column_letter(list(bt.columns).index("Result") + 1)
    ws.conditional_formatting.add(f"{res_col}{r0 + 1}:{res_col}{last2}", CellIsRule(
        operator="equal", formula=['"Rejected"'], fill=PatternFill("solid", fgColor="F4CCCC")))
    ws.conditional_formatting.add(f"{res_col}{r0 + 1}:{res_col}{last2}", CellIsRule(
        operator="equal", formula=['"Calibrated"'], fill=PatternFill("solid", fgColor="D9EAD3")))
    ws.cell(row=last2 + 2, column=1, value=(
        f"Backtest period {fc.index[0]:%Y-%m-%d} to {fc.index[-1]:%Y-%m-%d}. p-value < 0.05 = reject "
        "'model is calibrated'. Basel zone (99% only): Green 0-4, Yellow 5-9, Red 10+ breaches.")).font = NOTE_FONT
    _autowidth(ws, max_w=16)
    ws.column_dimensions["A"].width = 14
    return vt, bt


def sheet_returns(wb, prices):
    """Daily SIMPLE returns of the 5 stocks - the raw data the VBA macro reads."""
    ws = wb.create_sheet("Returns")
    simple = prices[config.EQUITIES].pct_change().dropna()
    _table(ws, simple, 1, fmts={c: "0.0000%" for c in config.EQUITIES}, index_label="Date")
    ws.freeze_panes = "B2"
    for col in "ABCDEF":
        ws.column_dimensions[col].width = 13
    return simple


def sheet_var_macro(wb, simple: pd.DataFrame, vt: pd.DataFrame):
    """Inputs for the VBA RefreshVaR macro + Python reference values to check it against.

    Cell map (the .bas file relies on these addresses - keep them in sync):
      C5:C9    weights of the 5 stocks (yellow)     C10 = sum check
      C12      look-back window (days)              C13 portfolio value (Rs)
      C14      confidence level 1                   C15 confidence level 2
      A19:G25  results table written by the macro (green)
      I19:L25  Python reference numbers (same method) and the difference
    """
    ws = wb.create_sheet("VaR Macro")
    _title(ws, "VaR calculator driven by VBA - edit the yellow cells, then run the RefreshVaR macro",
           "Without the macro imported, the green area stays empty. See vba/README_VBA.md.")
    ws["A4"], ws["B4"], ws["C4"] = "Input", "Stock", "Weight"
    for c in ("A4", "B4", "C4"):
        ws[c].fill, ws[c].font = HEADER_FILL, HEADER_FONT
    for i, name in enumerate(config.EQUITIES):
        r = 5 + i
        ws.cell(row=r, column=1, value=f"Weight {i + 1}")
        ws.cell(row=r, column=2, value=name)
        c = ws.cell(row=r, column=3, value=config.PORTFOLIO_WEIGHTS[name])
        c.fill, c.number_format, c.border = INPUT_FILL, PCT1, BOX
    ws["A10"], ws["C10"] = "Sum of weights", "=SUM(C5:C9)"
    ws["C10"].number_format = PCT1
    ws["D10"] = '=IF(ABS(C10-1)>0.0001,"Weights should sum to 100%","OK")'

    inputs = [("Window (days)", config.VAR_WINDOW, "0"), ("Portfolio value (Rs)", config.PORTFOLIO_VALUE_INR, INR),
              ("Confidence level 1", 0.95, PCT1), ("Confidence level 2", 0.99, PCT1)]
    for k, (label, val, fmt) in enumerate(inputs):
        r = 12 + k
        ws.cell(row=r, column=1, value=label)
        c = ws.cell(row=r, column=3, value=val)
        c.fill, c.number_format, c.border = INPUT_FILL, fmt, BOX
    dv = DataValidation(type="decimal", operator="between", formula1="0.5", formula2="0.9999",
                        showErrorMessage=True, errorTitle="Confidence level",
                        error="Enter a probability between 50% and 99.99%")
    ws.add_data_validation(dv)
    dv.add("C14:C15")
    dv2 = DataValidation(type="whole", operator="between", formula1="20", formula2=str(len(simple)),
                         showErrorMessage=True, error=f"Window must be 20 to {len(simple)} days")
    ws.add_data_validation(dv2)
    dv2.add("C12")

    ws["A17"] = "Results (written by the macro)"
    ws["A17"].font = Font(bold=True)
    ws["I17"] = "Python reference (default inputs)"
    ws["I17"].font = Font(bold=True)
    heads = ["Method", "Level", "VaR %", "ES %", "VaR Rs", "ES Rs", "As of"]
    for j, h in enumerate(heads):
        c = ws.cell(row=19, column=1 + j, value=h)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
    for r in range(20, 24):
        for j in range(7):
            ws.cell(row=r, column=1 + j).fill = OUTPUT_FILL

    ref = vt[vt["Method"].isin(["Historical", "Parametric"])].reset_index(drop=True)
    for j, h in enumerate(["Python VaR %", "Python ES %", "Diff VaR (bp)", "Diff ES (bp)"]):
        c = ws.cell(row=19, column=9 + j, value=h)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
    # macro writes rows in order: Hist L1, Param L1, Hist L2, Param L2
    order = [("Historical", "95%"), ("Parametric", "95%"), ("Historical", "99%"), ("Parametric", "99%")]
    for k, (m, lvl) in enumerate(order):
        r = 20 + k
        row = ref[(ref["Method"] == m) & (ref["Level"] == lvl)].iloc[0]
        ws.cell(row=r, column=9, value=float(row["VaR %"])).number_format = "0.000%"
        ws.cell(row=r, column=10, value=float(row["ES %"])).number_format = "0.000%"
        ws.cell(row=r, column=11, value=f'=IF(ISNUMBER(C{r}),(C{r}-I{r})*10000,"")').number_format = "0.00"
        ws.cell(row=r, column=12, value=f'=IF(ISNUMBER(D{r}),(D{r}-J{r})*10000,"")').number_format = "0.00"
    ws["A25"] = ("Historical VaR = -PERCENTILE.INC(returns, 1-c). Parametric VaR = z_c * stdev - mean, "
                 "where stdev of the portfolio series equals sqrt(w' Sigma w). With default inputs the "
                 "diff columns should read ~0.")
    ws["A25"].font = NOTE_FONT
    for col, w in zip("ABCDEFGHIJKL", (22, 12, 16, 10, 14, 14, 18, 3, 13, 12, 13, 12)):
        ws.column_dimensions[col].width = w


def sheet_greeks(wb):
    ws = wb.create_sheet("Greeks")
    S, T, r, sig = 100.0, 0.5, 0.05, 0.20
    _title(ws, "Black-Scholes prices and Greeks across strikes",
           f"S = {S:g}, T = {T:g} yr, r = {r:.0%}, sigma = {sig:.0%}, no dividends. "
           "Vega per 1 vol point, theta per calendar day, rho per 1% rate move.")
    rows = []
    for K in range(70, 135, 5):
        rec = {"Strike": K}
        for kind in ("call", "put"):
            g = options.bs_greeks(S, K, T, r, sig, kind)
            for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
                rec[f"{kind.title()} {name}"] = float(g[name])
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("Strike")
    last, lastc = _table(ws, df, 4, fmts={c: NUM4 for c in df.columns}, index_label="Strike")
    ws.conditional_formatting.add(f"C5:C{last}", ColorScaleRule(
        start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="2A78D6"))
    ws.cell(row=last + 2, column=1, value=(
        "Check: Call - Put = S - K e^(-rT) on every row (put-call parity); gamma and vega are identical "
        "for calls and puts.")).font = NOTE_FONT
    ws.freeze_panes = "B5"
    _autowidth(ws, max_w=13)
    return df


def sheet_pricer(wb):
    """Inputs + native Excel Black-Scholes formulas + the VBA UDFs, side by side."""
    ws = wb.create_sheet("Pricer")
    _title(ws, "Black-Scholes pricer - edit the yellow inputs",
           "Column C = native Excel formulas (work everywhere). Column D = VBA functions from "
           "MarketLens.bas (#NAME? until the module is imported). Column E = difference.")
    inputs = [("Spot S", 100), ("Strike K", 100), ("Time to expiry T (years)", 1),
              ("Risk-free rate r", 0.05), ("Volatility sigma", 0.20), ("Dividend yield q", 0),
              ("Type (call/put)", "call")]
    for i, (label, val) in enumerate(inputs):
        r = 4 + i
        ws.cell(row=r, column=1, value=label)
        c = ws.cell(row=r, column=2, value=val)
        c.fill, c.border = INPUT_FILL, BOX
        if label.startswith(("Risk", "Vol", "Div")):
            c.number_format = PCT2
    dv = DataValidation(type="list", formula1='"call,put"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add("B10")

    # helper cells: d1, d2 (native formulas). _xlfn. prefix = Excel 2010+ function names.
    ws["A12"], ws["B12"] = "d1", "=(LN(B4/B5)+(B7-B9+B8^2/2)*B6)/(B8*SQRT(B6))"
    ws["A13"], ws["B13"] = "d2", "=B12-B8*SQRT(B6)"
    for c in ("B12", "B13"):
        ws[c].number_format = "0.000000"

    N = "_xlfn.NORM.S.DIST"
    call = f'B10="call"'
    native = {
        "Price": f"=IF({call},B4*EXP(-B9*B6)*{N}(B12,TRUE)-B5*EXP(-B7*B6)*{N}(B13,TRUE),"
                 f"B5*EXP(-B7*B6)*{N}(-B13,TRUE)-B4*EXP(-B9*B6)*{N}(-B12,TRUE))",
        "Delta": f"=IF({call},EXP(-B9*B6)*{N}(B12,TRUE),-EXP(-B9*B6)*{N}(-B12,TRUE))",
        "Gamma": f"=EXP(-B9*B6)*{N}(B12,FALSE)/(B4*B8*SQRT(B6))",
        "Vega": f"=B4*EXP(-B9*B6)*{N}(B12,FALSE)*SQRT(B6)/100",
        "Theta": f"=(-B4*EXP(-B9*B6)*{N}(B12,FALSE)*B8/(2*SQRT(B6))"
                 f"+IF({call},-B7*B5*EXP(-B7*B6)*{N}(B13,TRUE)+B9*B4*EXP(-B9*B6)*{N}(B12,TRUE),"
                 f"B7*B5*EXP(-B7*B6)*{N}(-B13,TRUE)-B9*B4*EXP(-B9*B6)*{N}(-B12,TRUE)))/365",
        "Rho": f"=IF({call},B5*B6*EXP(-B7*B6)*{N}(B13,TRUE),-B5*B6*EXP(-B7*B6)*{N}(-B13,TRUE))/100",
    }
    args = "B4,B5,B6,B7,B8,B10,B9"
    for j, h in enumerate(["Output", "", "Excel formula", "VBA function", "Difference"]):
        c = ws.cell(row=15, column=1 + j, value=h or None)
        if h:
            c.fill, c.font = HEADER_FILL, HEADER_FONT
    for i, (name, f) in enumerate(native.items()):
        r = 16 + i
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=3, value=f).number_format = "0.000000"
        udf = f"=BS_PRICE({args})" if name == "Price" else f'=BS_GREEK("{name.lower()}",{args})'
        ws.cell(row=r, column=4, value=udf).number_format = "0.000000"
        ws.cell(row=r, column=5, value=f'=IFERROR(D{r}-C{r},"import VBA")').number_format = "0.0E+00"

    ws["A23"] = "Implied vol from a price"
    ws["A23"].font = Font(bold=True)
    ws["A24"], ws["B24"] = "Market price", 10.4506
    ws["B24"].fill = INPUT_FILL
    ws["A25"], ws["D25"] = "Implied vol (VBA)", f"=BS_IMPLIED_VOL(B24,B4,B5,B6,B7,B10,B9)"
    ws["D25"].number_format = PCT2
    ws["A27"] = ("Python reference for the default inputs: call 10.450584, delta 0.636831, gamma 0.018762, "
                 "vega 0.375240, theta -0.017573, rho 0.532325; IV of 10.4506 = 20.00%.")
    ws["A27"].font = NOTE_FONT
    for col, w in zip("ABCDE", (26, 12, 16, 16, 14)):
        ws.column_dimensions[col].width = w


def sheet_pairs(wb, prices):
    ws = wb.create_sheet("Pairs")
    Y, X = config.PAIR
    start = prices.index[config.FORMATION_DAYS]
    _title(ws, f"Pairs trading {Y} / {X} - out-of-sample {start:%b %Y} to {prices.index[-1]:%b %Y}",
           f"Entry |z|>{config.Z_ENTRY}, exit |z|<{config.Z_EXIT}, stop |z|>{config.Z_STOP}, "
           f"{config.COST_BPS} bp per leg, 1-day execution lag. Sharpe with rf = 0 for all lines.")
    perf, trades_ols = {}, None
    for method, label in (("ols", "Pairs - rolling EG hedge"), ("ratio", "Pairs - 1:1 log-ratio")):
        bt = strategy.backtest(prices[Y], prices[X], method=method)
        r = bt.loc[start:, "net_ret"]
        t = strategy.trade_log(bt)
        t = t[t["entry"] >= start]
        perf[label] = strategy.metrics(r, trades=t)
        if method == "ols":
            trades_ols = t
    simple = prices.loc[start:].pct_change().fillna(0)
    bh = (simple[[Y, X]] + 1).cumprod().mean(axis=1).pct_change().fillna(0)
    perf[f"Buy & hold 50/50 {Y}+{X}"] = strategy.metrics(bh)
    perf["Buy & hold NIFTY 50"] = strategy.metrics(simple["NIFTY50"])
    pdf = pd.DataFrame(perf).T
    fm = {c: PCT2 for c in ("Total return", "CAGR", "Volatility", "Max drawdown", "Win rate", "Avg trade")}
    fm.update({"Sharpe": NUM2, "Trades": "0", "Avg holding days": "0.0"})
    last, _ = _table(ws, pdf, 4, fmts=fm, index_label="Strategy")

    eg = strategy.engle_granger(prices.iloc[: config.FORMATION_DAYS][Y], prices.iloc[: config.FORMATION_DAYS][X])
    ws.cell(row=last + 2, column=1, value=(
        f"Engle-Granger (formation, first {config.FORMATION_DAYS} days): t = {eg['t_stat']:.2f} vs 5% critical "
        f"{eg['crit_5pct']:.2f}, p = {eg['p_value']:.3f} -> NOT cointegrated at 5%.")).font = NOTE_FONT

    r0 = last + 5
    ws.cell(row=r0 - 1, column=1, value="Trade log - rolling EG hedge ratio").font = Font(bold=True)
    last2, _ = _table(ws, trades_ols, r0, index=False, fmts={"return": PCT2})
    ret_col = get_column_letter(list(trades_ols.columns).index("return") + 1)
    ws.conditional_formatting.add(f"{ret_col}{r0 + 1}:{ret_col}{last2}", CellIsRule(
        operator="lessThan", formula=["0"], font=Font(color="C00000")))
    _autowidth(ws, max_w=30)
    return pdf


def sheet_summary(wb, prices, summ, vt, bt, pdf, mc_rows):
    ws = wb["Summary"]
    _title(ws, "MarketLens - Markets Risk & Pricing Toolkit",
           f"Data: Yahoo Finance via yfinance, {prices.index[0]:%d %b %Y} - {prices.index[-1]:%d %b %Y}. "
           "All numbers computed by the Python package in this repo.")
    v99 = vt[(vt["Method"] == "Historical") & (vt["Level"] == "99%")].iloc[0]
    b99 = bt[(bt["Method"] == "Historical") & (bt["Level"] == "99%")].iloc[0]
    b95 = bt[(bt["Method"] == "Historical") & (bt["Level"] == "95%")].iloc[0]
    mc100k = next(r for r in mc_rows if r["paths"] == 100_000)
    pairs = pdf.loc["Pairs - rolling EG hedge"]
    items = [
        ("Risk", "NIFTY 50 annualised volatility", summ.loc["NIFTY50", "Ann. volatility"], PCT2),
        ("Risk", "Best Sharpe (ICICIBANK)", summ.loc["ICICIBANK", "Sharpe"], NUM2),
        ("Risk", "Worst max drawdown (TCS)", summ.loc["TCS", "Max drawdown"], PCT2),
        ("VaR", "1-day 99% historical VaR (Rs)", v99["VaR INR"], INR),
        ("VaR", "1-day 99% historical ES (Rs)", v99["ES INR"], INR),
        ("VaR", "99% VaR breaches, last 250 days (expected 2.5)", b99["Breaches"], "0"),
        ("VaR", "Kupiec p-value, 99% historical", b99["p-value"], "0.000"),
        ("VaR", "95% VaR breaches, last 250 days (expected 12.5)", b95["Breaches"], "0"),
        ("VaR", "Kupiec p-value, 95% historical", b95["p-value"], "0.000"),
        ("Options", "Black-Scholes ATM call (S=K=100, T=1, r=5%, vol=20%)", mc100k["bs_price"], NUM4),
        ("Options", "Monte Carlo mean abs error at 100k paths", mc100k["mean_abs_error_pct"], "0.000%"),
        ("Pairs", "Pairs (rolling EG) total return, out-of-sample", pairs["Total return"], PCT2),
        ("Pairs", "Pairs (rolling EG) Sharpe", pairs["Sharpe"], NUM2),
        ("Pairs", "Pairs (rolling EG) win rate", pairs["Win rate"], PCT1),
    ]
    for j, h in enumerate(["Area", "Metric", "Value"]):
        c = ws.cell(row=4, column=1 + j, value=h)
        c.fill, c.font, c.border = HEADER_FILL, HEADER_FONT, BOX
    for i, (area, name, val, fmt) in enumerate(items, start=5):
        ws.cell(row=i, column=1, value=area).border = BOX
        ws.cell(row=i, column=2, value=name).border = BOX
        c = ws.cell(row=i, column=3, value=float(val))
        c.number_format, c.border = fmt, BOX
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 56
    ws.column_dimensions["C"].width = 16
    ws.cell(row=len(items) + 7, column=1, value="Legend: yellow cells = inputs you can edit; green cells = "
            "written by the VBA macro.").font = NOTE_FONT


# ===========================================================================
def build_workbook(path=XLSX_PATH):
    prices, rets = load_data()
    wb = Workbook()
    wb.active.title = "Summary"
    summ = sheet_risk(wb, prices, rets)
    vt, bt = sheet_var(wb, rets)
    simple = sheet_returns(wb, prices)
    sheet_var_macro(wb, simple, vt)
    sheet_greeks(wb)
    sheet_pricer(wb)
    pdf = sheet_pairs(wb, prices)
    mc_rows = options.mc_convergence(**config.EXAMPLE_OPTION, path_counts=(100_000,))
    sheet_summary(wb, prices, summ, vt, bt, pdf, mc_rows)
    # sheet order: Summary, Risk, VaR, VaR Macro, Greeks, Pricer, Pairs, Returns
    wb._sheets = [wb[n] for n in ("Summary", "Risk", "VaR", "VaR Macro", "Greeks", "Pricer", "Pairs", "Returns")]
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = ws.title in ("Returns",)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


if __name__ == "__main__":
    print("Saved", build_workbook())
