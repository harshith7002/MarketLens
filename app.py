"""MarketLens dashboard.

Run:  streamlit run app.py

Four tabs, each a thin UI over the tested library code in marketlens/:
  Risk           -> marketlens.risk      (metrics, correlation)
  VaR            -> marketlens.risk      (VaR / ES, backtest, Kupiec)
  Options        -> marketlens.options   (Black-Scholes, Greeks, MC, IV, hedging)
  Pairs Trading  -> marketlens.strategy  (cointegration, z-score backtest)
The app does no maths of its own, so what you see is what the tests check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from marketlens import config, options, risk, strategy
from marketlens.data import load_data, load_option_chain

st.set_page_config(page_title="MarketLens", page_icon="📈", layout="wide")

# Validated categorical palette (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
LAYOUT = dict(margin=dict(l=10, r=10, t=75, b=10), hovermode="x unified",  # room for title + legend
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def pct(x, d=2):
    return f"{x:.{d}%}"


# ---------------------------------------------------------------------------
# Cached data and heavy computations
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_chain():
    return load_option_chain()


@st.cache_data(show_spinner="Running the rolling VaR backtest...")
def get_backtest(weights: tuple, window: int, test_days: int, n_sims: int):
    _, rets = get_data()
    fc = risk.rolling_var_forecasts(rets, dict(weights), config.VAR_LEVELS, window, test_days, n_sims)
    return fc, risk.backtest_summary(fc)


@st.cache_data(show_spinner="Simulating paths...")
def get_mc_convergence(S, K, T, r, sigma, kind):
    return pd.DataFrame(options.mc_convergence(S, K, T, r, sigma, kind, n_seeds=10))


@st.cache_data(show_spinner="Simulating hedges...")
def get_hedge_sim(sig_true, sig_hedge, step, cost, n_paths, days, mu):
    rng = np.random.default_rng(config.SEED)
    T = days / 252
    paths = options.gbm_paths(100, T, 0.05, sig_true, days, n_paths, rng, mu=mu)
    return options.delta_hedge(paths[::step], 100, T, 0.05, sig_hedge, cost_bps=cost)


prices, rets = get_data()
st.title("MarketLens — Markets Risk & Pricing Toolkit")
st.caption(f"Daily data from Yahoo Finance, {prices.index[0]:%d %b %Y} – {prices.index[-1]:%d %b %Y}, "
           f"{len(prices):,} NSE trading days, cached locally. Every number below is computed live by the "
           "`marketlens` package.")

tab_risk, tab_var, tab_opt, tab_pairs = st.tabs(["Risk", "VaR", "Options", "Pairs Trading"])

# ===========================================================================
# RISK
# ===========================================================================
with tab_risk:
    st.subheader("Per-asset risk and return")
    summ = risk.asset_summary(prices, rets)
    st.dataframe(summ.style.format({"Ann. return": pct, "Ann. volatility": pct, "Sharpe": "{:.2f}",
                                    "Max drawdown": pct}))
    st.caption(f"Return = CAGR from log returns; volatility = daily σ × √252; Sharpe uses "
               f"rf = {config.RISK_FREE_RATE:.0%} (assumption, set in config.py).")

    c1, c2 = st.columns([1, 1])
    with c1:
        corr = risk.correlation_matrix(rets)
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                                   text=corr.round(2).values, texttemplate="%{text}",
                                   hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_layout(title="Correlation of daily log returns", height=460,
                          margin=dict(l=10, r=10, t=40, b=10), yaxis_autorange="reversed")
        st.plotly_chart(fig)
    with c2:
        names = list(prices.columns)
        a = st.selectbox("Asset A", names, index=names.index("NIFTY50"))
        b = st.selectbox("Asset B", names, index=names.index("USDINR"))
        win = st.slider("Rolling window (days)", 20, 250, config.ROLLING_CORR_WINDOW, step=10)
        if a == b:
            st.info("Pick two different assets.")
        else:
            rc = risk.rolling_correlation(rets, a, b, win)
            fig = go.Figure(go.Scatter(x=rc.index, y=rc.values, line=dict(color=SERIES[0], width=2),
                                       name=f"{win}-day correlation"))
            fig.add_hline(y=corr.loc[a, b], line=dict(color=GREY, dash="dash"),
                          annotation_text=f"full-period {corr.loc[a, b]:.2f}")
            fig.add_hline(y=0, line=dict(color=GREY, width=1))
            fig.update_layout(title=f"Rolling correlation: {a} vs {b}", height=330,
                              yaxis_title="Correlation", **LAYOUT)
            st.plotly_chart(fig)

    st.subheader("Growth and drawdowns")
    pick = st.multiselect("Assets", names, default=["NIFTY50", "ICICIBANK", "TCS", "GOLD"], max_selections=8)
    if pick:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
        for i, n in enumerate(pick):
            col = SERIES[names.index(n) % len(SERIES)]
            fig.add_trace(go.Scatter(x=prices.index, y=prices[n] / prices[n].iloc[0] * 100, name=n,
                                     line=dict(color=col, width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=prices.index, y=risk.drawdown_series(prices[n]) * 100, name=n,
                                     line=dict(color=col, width=1.4), showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text="Rebased to 100", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_layout(height=520, **LAYOUT)
        st.plotly_chart(fig)

# ===========================================================================
# VaR
# ===========================================================================
with tab_var:
    st.subheader("Portfolio Value-at-Risk and Expected Shortfall (1-day)")
    left, right = st.columns([1, 2.4])
    with left:
        st.markdown("**Weights**")
        raw = {n: st.number_input(n, 0.0, 1.0, config.PORTFOLIO_WEIGHTS[n], 0.05, key=f"w_{n}")
               for n in config.EQUITIES}
        total = sum(raw.values())
        if total <= 0:
            st.error("At least one weight must be positive.")
            st.stop()
        weights = {k: v / total for k, v in raw.items()}
        if abs(total - 1) > 1e-9:
            st.caption(f"Weights sum to {total:.0%}; rescaled to 100%.")
        window = st.slider("Look-back window (days)", 60, 1000, config.VAR_WINDOW, step=10)
        notional = st.number_input("Portfolio value (Rs)", 100_000, 10_000_000_000,
                                   config.PORTFOLIO_VALUE_INR, step=1_000_000, format="%d")

    with right:
        vt = risk.var_table(rets, weights, config.VAR_LEVELS, window, notional)
        k1, k2, k3, k4 = st.columns(4)
        h99 = vt[(vt.Method == "Historical") & (vt.Level == "99%")].iloc[0]
        p99 = vt[(vt.Method == "Parametric") & (vt.Level == "99%")].iloc[0]
        k1.metric("99% VaR, historical", f"Rs {h99['VaR INR']:,.0f}", pct(h99["VaR %"], 3), delta_color="off", delta_arrow="off")
        k2.metric("99% ES, historical", f"Rs {h99['ES INR']:,.0f}", pct(h99["ES %"], 3), delta_color="off", delta_arrow="off")
        k3.metric("99% VaR, parametric", f"Rs {p99['VaR INR']:,.0f}", pct(p99["VaR %"], 3), delta_color="off", delta_arrow="off")
        k4.metric("ES / VaR (hist. 99%)", f"{h99['ES INR'] / h99['VaR INR']:.2f}×")
        st.dataframe(vt.style.format({"VaR %": lambda v: pct(v, 3), "ES %": lambda v: pct(v, 3),
                                      "VaR INR": "Rs {:,.0f}", "ES INR": "Rs {:,.0f}"}), hide_index=True)

        port = risk.portfolio_returns(rets.iloc[-window:], weights)
        fig = go.Figure(go.Histogram(x=port * 100, nbinsx=60, marker_color=SERIES[0], opacity=0.75,
                                     name="Daily returns"))
        for j, (m, dash) in enumerate((("Historical", "solid"), ("Parametric", "dash"))):
            for lvl in ("95%", "99%"):
                v = vt[(vt.Method == m) & (vt.Level == lvl)]["VaR %"].iloc[0]
                fig.add_vline(x=-v * 100, line=dict(color=SERIES[1 if lvl == "95%" else 7], dash=dash, width=1.5),
                              annotation_text=f"{'Hist' if m == 'Historical' else 'Param'} {lvl}",
                              # historical and parametric lines sit close together: label them on
                              # opposite sides so the text never overlaps
                              annotation_position="top right" if m == "Historical" else "top left",
                              annotation_font_size=10)
        fig.update_layout(title=f"Portfolio daily returns, last {window} days, with -VaR lines",
                          xaxis_title="Daily return (%)", yaxis_title="Days", height=360,
                          margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig)

    st.subheader("Backtest: did the model's VaR hold up out-of-sample?")
    st.caption("Each day's VaR uses only the previous window of returns, then is compared with that day's "
               "realised loss. Kupiec test H0: breach rate = 1 − confidence level.")
    b1, b2 = st.columns([1, 3])
    with b1:
        bt_window = st.select_slider("Estimation window", [125, 250, 500], value=250)
        bt_days = st.select_slider("Days to backtest", [125, 250, 500], value=250)
        bt_level = st.radio("Chart level", ["99", "95"], horizontal=True)
        st.caption(f"Monte Carlo uses {config.MC_SIMS:,} simulations per day. The first run of a new "
                   "setting takes ~20 s, then it is cached. (With only 2,000 simulations the 99% "
                   "Monte Carlo verdict flips between runs - tail quantiles need many draws.)")
    if bt_window + bt_days > len(rets):
        st.error("Not enough history for that combination.")
    else:
        fc, bts = get_backtest(tuple(weights.items()), bt_window, bt_days, config.MC_SIMS)
        with b2:
            st.dataframe(bts.style.format({"Expected": "{:.1f}", "Kupiec LR": "{:.3f}", "p-value": "{:.3f}"})
                         .map(lambda v: "background-color:#f4cccc" if v == "Rejected" else
                              ("background-color:#d9ead3" if v == "Calibrated" else ""), subset=["Result"]),
                         hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=fc.index, y=fc["Realised"] * 100, marker_color="#c9c8c2", name="Realised return"))
        for i, m in enumerate(["Historical", "Parametric", "Monte Carlo"]):
            fig.add_trace(go.Scatter(x=fc.index, y=-fc[f"{m}_{bt_level}"] * 100, name=f"−VaR {bt_level}% {m}",
                                     line=dict(color=SERIES[i], width=1.8)))
        hit = fc["Realised"] < -fc[f"Historical_{bt_level}"]
        fig.add_trace(go.Scatter(x=fc.index[hit], y=fc["Realised"][hit] * 100, mode="markers",
                                 marker=dict(color=SERIES[7], size=9, line=dict(color="white", width=1.5)),
                                 name="Breach (historical)"))
        fig.update_layout(height=400, yaxis_title="Daily return (%)", **LAYOUT)
        st.plotly_chart(fig)

# ===========================================================================
# OPTIONS
# ===========================================================================
with tab_opt:
    st.subheader("Black-Scholes pricer and Greeks")
    c = st.columns(7)
    S = c[0].number_input("Spot S", 1.0, 1e6, 100.0)
    K = c[1].number_input("Strike K", 1.0, 1e6, 100.0)
    T = c[2].number_input("Years to expiry T", 0.01, 10.0, 1.0, 0.05)
    r = c[3].number_input("Rate r (%)", -5.0, 30.0, 5.0, 0.25) / 100
    sig = c[4].number_input("Volatility σ (%)", 1.0, 300.0, 20.0, 1.0) / 100
    q = c[5].number_input("Dividend yield q (%)", 0.0, 20.0, 0.0, 0.25) / 100
    kind = c[6].radio("Type", ["call", "put"], horizontal=True)

    g = {k: float(v) for k, v in options.bs_greeks(S, K, T, r, sig, kind, q).items()}
    other = float(options.bs_price(S, K, T, r, sig, "put" if kind == "call" else "call", q))
    m = st.columns(6)
    for col, (name, label) in zip(m, [("price", "Price"), ("delta", "Delta"), ("gamma", "Gamma"),
                                      ("vega", "Vega / vol pt"), ("theta", "Theta / day"), ("rho", "Rho / 1%")]):
        col.metric(label, f"{g[name]:.4f}")
    call_p, put_p = (g["price"], other) if kind == "call" else (other, g["price"])
    st.caption(f"Put-call parity check: C − P = {call_p - put_p:.6f}; "
               f"S·e^(−qT) − K·e^(−rT) = {S * np.exp(-q * T) - K * np.exp(-r * T):.6f}")

    spots = np.linspace(0.5 * K, 1.5 * K, 201)
    gs = options.bs_greeks(spots, K, T, r, sig, kind, q)
    fig = make_subplots(rows=2, cols=3, subplot_titles=["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"],
                        vertical_spacing=0.14)
    for i, name in enumerate(["price", "delta", "gamma", "vega", "theta", "rho"]):
        fig.add_trace(go.Scatter(x=spots, y=gs[name], line=dict(color=SERIES[0], width=2), name=name,
                                 showlegend=False), row=i // 3 + 1, col=i % 3 + 1)
        fig.add_vline(x=S, line=dict(color=GREY, dash="dot", width=1), row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(height=490, margin=dict(l=10, r=10, t=80, b=10), title="Greeks vs spot (dotted = current spot)")
    st.plotly_chart(fig)

    o1, o2 = st.columns(2)
    with o1:
        st.subheader("Monte Carlo convergence")
        conv = get_mc_convergence(S, K, T, r, sig, kind)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conv["paths"], y=conv["mean_abs_error"], mode="lines+markers",
                                 name="Mean |MC − BS| (10 seeds)", line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=conv["paths"], y=conv["mean_std_error"], mode="lines+markers",
                                 name="Standard error", line=dict(color=SERIES[1], width=2, dash="dash")))
        ref = conv["mean_abs_error"].iloc[0] * np.sqrt(conv["paths"].iloc[0] / conv["paths"])
        fig.add_trace(go.Scatter(x=conv["paths"], y=ref, name="1/√N", line=dict(color=GREY, dash="dot")))
        fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Paths", yaxis_title="Error",
                          height=380, **{**LAYOUT, "hovermode": "closest"})
        st.plotly_chart(fig)
        row = conv[conv["paths"] == 100_000].iloc[0]
        st.caption(f"At 100,000 paths the average error is {pct(row['mean_abs_error_pct'], 3)} of the "
                   f"Black-Scholes price {row['bs_price']:.4f}.")

    with o2:
        st.subheader("Implied volatility solver")
        mkt = st.number_input("Market price of this option", 0.0001, 1e6, round(g["price"] * 1.05, 4), format="%.4f")
        res = options.implied_vol(mkt, S, K, T, r, kind, q)
        if np.isnan(res.sigma):
            st.warning(f"No implied vol: {res.method}.")
        else:
            st.metric("Implied volatility", pct(res.sigma), f"{res.method}, {res.iterations} iterations",
                      delta_color="off", delta_arrow="off")
        chain = get_chain()
        exp_, qd = chain["expiry"].iloc[0], chain["quote_date"].iloc[0]
        Tc = (pd.Timestamp(exp_) - pd.Timestamp(qd)).days / 365
        fwd = options.implied_forward(chain, Tc, config.USD_RISK_FREE_RATE)
        smile = options.iv_smile(chain, Tc, config.USD_RISK_FREE_RATE, fwd["q"])
        fig = go.Figure()
        for i, k_ in enumerate(["put", "call"]):
            d = smile[smile["type"] == k_]
            fig.add_trace(go.Scatter(x=d["moneyness"], y=d["iv"] * 100, mode="markers", name=f"OTM {k_}s",
                                     marker=dict(color=SERIES[i], size=6),
                                     customdata=d[["strike", "mid"]],
                                     hovertemplate="K=%{customdata[0]} mid=%{customdata[1]:.2f} IV=%{y:.2f}%<extra></extra>"))
        fig.update_layout(title=f"Real SPY skew (quotes {qd}, expiry {exp_})", xaxis_title="K / Forward",
                          yaxis_title="Implied vol (%)", height=330, **{**LAYOUT, "hovermode": "closest"})
        st.plotly_chart(fig)

    st.subheader("Delta-hedging a short call (simulated)")
    h1, h2 = st.columns([1, 2.4])
    with h1:
        sig_hedge = st.slider("Vol the call was sold at (%)", 5, 60, 20) / 100
        sig_true = st.slider("Vol the stock actually realises (%)", 5, 60, 20) / 100
        freq = st.radio("Rebalance", ["Daily", "Weekly", "Monthly"], horizontal=True)
        cost = st.slider("Transaction cost (bp)", 0, 30, 0)
        mu = st.slider("Real-world drift μ (%)", -20, 40, 12) / 100
    step = {"Daily": 1, "Weekly": 5, "Monthly": 20}[freq]
    hd = get_hedge_sim(sig_true, sig_hedge, step, cost, 10_000, 60, mu)
    with h2:
        pnl, prem = hd["pnl"], float(np.mean(hd["premium"]))
        mm = st.columns(3)
        mm[0].metric("Premium received", f"{prem:.3f}")
        mm[1].metric("Mean hedged P&L", f"{pnl.mean():+.3f}")
        mm[2].metric("Std of P&L", f"{pnl.std():.3f}", f"{pnl.std() / prem:.1%} of premium", delta_color="off", delta_arrow="off")
        fig = go.Figure(go.Histogram(x=pnl, nbinsx=80, marker_color=SERIES[2], opacity=0.8))
        fig.add_vline(x=0, line=dict(color=GREY))
        fig.update_layout(title=f"P&L at expiry, 10,000 paths, 3-month ATM call, {freq.lower()} hedge",
                          xaxis_title="P&L per option", yaxis_title="Paths", height=330,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)
    st.caption("Try: realised vol above the sold vol → the hedger loses about the vega difference. "
               "Change μ → the mean P&L barely moves, because the hedge removes the drift (the core Black-Scholes idea).")

# ===========================================================================
# PAIRS TRADING
# ===========================================================================
with tab_pairs:
    st.subheader("Pairs trading: cointegration and z-score backtest")
    p1, p2 = st.columns([1, 2.6])
    with p1:
        Y = st.selectbox("Stock Y (traded vs X)", config.EQUITIES, index=config.EQUITIES.index(config.PAIR[0]))
        X = st.selectbox("Stock X", config.EQUITIES, index=config.EQUITIES.index(config.PAIR[1]))
        method = st.radio("Hedge ratio", ["ols", "ratio"], horizontal=True,
                          format_func=lambda m: "Rolling Engle-Granger β" if m == "ols" else "1:1 log-ratio")
        z_in = st.slider("Entry |z|", 1.0, 3.5, config.Z_ENTRY, 0.25)
        z_out = st.slider("Exit |z|", 0.0, 1.5, config.Z_EXIT, 0.25)
        z_stop = st.slider("Stop-loss |z|", 3.0, 6.0, config.Z_STOP, 0.5)
        cost_bp = st.slider("Cost per leg (bp)", 0, 30, config.COST_BPS)
        st.caption("Defaults are fixed textbook values. Tuning these sliders to the best result on this "
                   "same data is overfitting — the out-of-sample numbers would no longer be honest.")
    if Y == X:
        p2.info("Pick two different stocks.")
    else:
        start = prices.index[config.FORMATION_DAYS]
        form = prices.iloc[: config.FORMATION_DAYS]
        egs = {lab: strategy.engle_granger(df[Y], df[X]) for lab, df in
               (("Formation (first 2y)", form), ("Trading (last 3y)", prices.loc[start:]), ("Full 5y", prices))}
        bt = strategy.backtest(prices[Y], prices[X], method, cost_bp, config.EXEC_LAG, z_in, z_out, z_stop)
        trades = strategy.trade_log(bt)
        trades = trades[trades["entry"] >= start] if len(trades) else trades
        r = bt.loc[start:, "net_ret"]
        mstat = strategy.metrics(r, trades=trades)
        simple = prices.loc[start:].pct_change().fillna(0)
        bh = (simple[[Y, X]] + 1).cumprod().mean(axis=1).pct_change().fillna(0)
        mb, mn = strategy.metrics(bh), strategy.metrics(simple["NIFTY50"])
        with p2:
            fe = egs["Formation (first 2y)"]
            verdict = "cointegrated" if fe["p_value"] < 0.05 else "NOT cointegrated"
            (st.success if fe["p_value"] < 0.05 else st.warning)(
                f"Engle-Granger on the formation period: t = {fe['t_stat']:.2f} vs 5% critical "
                f"{fe['crit_5pct']:.2f}, p = {fe['p_value']:.3f} → **{verdict}** at 5%. "
                f"Return correlation {np.log(prices[[Y, X]]).diff().corr().iloc[0, 1]:.2f}.")
            st.dataframe(pd.DataFrame(egs).T[["beta", "t_stat", "crit_5pct", "p_value", "half_life_days"]]
                         .style.format("{:.3f}"))
            k = st.columns(5)
            k[0].metric("Total return", pct(mstat["Total return"]), f"B&H pair {pct(mb['Total return'])}",
                        delta_color="off", delta_arrow="off")
            k[1].metric("Sharpe (rf=0)", f"{mstat['Sharpe']:.2f}", f"NIFTY {mn['Sharpe']:.2f}", delta_color="off", delta_arrow="off")
            k[2].metric("Max drawdown", pct(mstat["Max drawdown"]), f"B&H pair {pct(mb['Max drawdown'])}",
                        delta_color="off", delta_arrow="off")
            k[3].metric("Win rate", pct(mstat["Win rate"], 1) if mstat["Trades"] else "–")
            k[4].metric("Trades", f"{mstat['Trades']}")

        seg = bt.loc[start:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=seg.index, y=seg["z"], line=dict(color=GREY, width=1.2), name="z-score"),
                      row=1, col=1)
        for lvl, dash in ((z_in, "dash"), (-z_in, "dash"), (z_out, "dot"), (-z_out, "dot")):
            fig.add_hline(y=lvl, line=dict(color=GREY, dash=dash, width=1), row=1, col=1)
        for side, colr, name in ((1, SERIES[2], "Long spread"), (-1, SERIES[7], "Short spread")):
            mask = seg["position"] == side
            fig.add_trace(go.Scatter(x=seg.index[mask], y=seg["z"][mask], mode="markers", name=name,
                                     marker=dict(color=colr, size=4)), row=1, col=1)
        curves = {"Pairs strategy": (1 + r).cumprod(), f"Buy & hold {Y}+{X}": (1 + bh).cumprod(),
                  "Buy & hold NIFTY 50": (1 + simple["NIFTY50"]).cumprod()}
        for i, (lab, eq) in enumerate(curves.items()):
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=lab, line=dict(color=SERIES[i], width=2)),
                          row=2, col=1)
        fig.update_yaxes(title_text="Spread z-score", row=1, col=1)
        fig.update_yaxes(title_text="Growth of Rs 1", row=2, col=1)
        fig.update_layout(height=620, title=f"Out-of-sample {start:%b %Y} – {prices.index[-1]:%b %Y}, after costs",
                          **LAYOUT)
        st.plotly_chart(fig)
        with st.expander(f"Trade log ({len(trades)} trades)"):
            st.dataframe(trades.style.format({"return": pct, "entry": "{:%Y-%m-%d}", "exit": "{:%Y-%m-%d}"}),
                         hide_index=True)
