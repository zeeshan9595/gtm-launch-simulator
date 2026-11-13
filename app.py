"""Streamlit workbench for the GTM launch simulator. Run with: streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data.generate_ecosystem import generate_users
from engine import experiment as ex
from engine import unit_economics as ue
from engine.memo import write_memo

st.set_page_config(page_title="GTM Launch Simulator", layout="wide")
BLUE, GREY, GREEN, RED = "#2563EB", "#94A3B8", "#15803D", "#B91C1C"


@st.cache_data
def load_users(n: int, seed: int):
    return generate_users(n, seed)


st.title("GTM Launch Simulator")
st.caption("Cashback debit card launch across a simulated multi-product customer base. "
           "Synthetic data and illustrative assumptions throughout.")

with st.sidebar:
    st.header("Launch")
    cashback = st.slider("Cashback rate (%)", 0.0, 2.5, 1.0, 0.1) / 100
    n_users = st.select_slider("Simulated base", [20_000, 50_000, 100_000], 100_000)
    st.header("Assumptions")
    interchange = st.slider("Interchange rate (%)", 0.2, 2.0, 1.2, 0.1) / 100
    cannibal = st.slider("Cannibalisation of existing fees (%)", 0, 50, 15, 5) / 100
    halo = st.slider("Deposit balance uplift (%)", 0, 20, 8, 1) / 100
    churn = st.slider("Monthly card churn (%)", 0.0, 10.0, 3.0, 0.5) / 100
    cap = st.slider("Cashback cap per month", 1.0, 50.0, 15.0, 1.0)
    st.header("Launch test")
    mde = st.slider("Minimum detectable lift (points)", 0.5, 5.0, 2.0, 0.5) / 100

users = load_users(n_users, 42)
a = ue.Assumptions(interchange_rate=interchange, cannibalisation_rate=cannibal, deposit_halo=halo,
                   monthly_card_churn=churn, cashback_cap=cap)
pnl = ue.launch_pnl(users, cashback, a)
sweep = ue.sweep_cashback(users, np.round(np.arange(0, 0.0251, 0.001), 4), a)
best = ue.best_cashback(sweep)
baseline = float(ue.adoption_probability(users, 0.0).mean())
needed = ex.sample_size_per_group(baseline, mde)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Expected adopters", f"{pnl['adopters']:,.0f}", f"{pnl['adoption_rate']:.1%} of base", delta_color="off")
k2.metric("12-month contribution", f"{pnl['contribution']:,.0f}")
k3.metric("Payback", f"Month {pnl['payback_month']}" if pnl["payback_month"] else "Not reached")
k4.metric("LTV to CAC", f"{pnl['ltv_to_cac']:.1f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pricing curve", "12-month P&L", "Segments", "Launch test", "Decision memo"])

with tab1:
    st.subheader(f"Contribution peaks at {best:.1%} cashback")
    left, right = st.columns(2)
    fig = go.Figure(go.Scatter(x=sweep["cashback_rate"] * 100, y=sweep["adoption_rate"] * 100,
                               mode="lines", line=dict(color=BLUE, width=2), name="Adoption",
                               hovertemplate="%{x:.1f}% cashback<br>%{y:.1f}% adoption<extra></extra>"))
    fig.add_vline(x=cashback * 100, line_dash="dot", line_color=GREY, annotation_text="Selected")
    fig.update_layout(title="Adoption by cashback rate", xaxis_title="Cashback rate (%)",
                      yaxis_title="Adoption (% of base)", height=380, margin=dict(t=50, b=40))
    left.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=sweep["cashback_rate"] * 100, y=sweep["contribution"],
                               mode="lines", line=dict(color=BLUE, width=2), name="Contribution",
                               hovertemplate="%{x:.1f}% cashback<br>%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.add_vline(x=cashback * 100, line_dash="dot", line_color=GREY, annotation_text="Selected")
    fig.update_layout(title="12-month contribution by cashback rate", xaxis_title="Cashback rate (%)",
                      yaxis_title="Contribution", height=380, margin=dict(t=50, b=40))
    right.plotly_chart(fig, width="stretch")

with tab2:
    c = pnl["components"]
    labels = ["Interchange", "Deposit halo", "Cashback", "Cannibalisation", "Servicing", "Issuance", "Acquisition"]
    values = [c["interchange"], c["deposit_halo"], c["cashback"], c["cannibalisation"],
              c["servicing"], c["issuance"], c["acquisition"]]
    fig = go.Figure(go.Waterfall(
        x=labels + ["Contribution"], y=values + [0], measure=["relative"] * len(values) + ["total"],
        increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED), totals=dict(marker_color=BLUE),
        connector=dict(line=dict(color=GREY, width=1)), hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="What builds and erodes contribution", yaxis_title="Amount over the horizon",
                      height=420, showlegend=False, margin=dict(t=50, b=40))
    st.plotly_chart(fig, width="stretch")
    months = list(range(1, len(pnl["cumulative_by_month"]) + 1))
    fig = go.Figure(go.Scatter(x=months, y=pnl["cumulative_by_month"], mode="lines+markers",
                               line=dict(color=BLUE, width=2), marker=dict(size=8),
                               hovertemplate="Month %{x}<br>%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.update_layout(title="Cumulative contribution of the launch cohort", xaxis_title="Month",
                      yaxis_title="Cumulative contribution", height=340, margin=dict(t=50, b=40))
    st.plotly_chart(fig, width="stretch")

with tab3:
    seg = ue.segment_breakdown(users, cashback, a)
    fig = go.Figure([
        go.Bar(name="Deposit halo", x=seg["segment"], y=seg["deposit_halo"], marker_color=GREEN,
               hovertemplate="%{x}<br>Halo %{y:,.0f}<extra></extra>"),
        go.Bar(name="Cannibalisation", x=seg["segment"], y=seg["cannibalisation"], marker_color=RED,
               hovertemplate="%{x}<br>Cannibalisation %{y:,.0f}<extra></extra>"),
    ])
    fig.update_layout(title="Cannibalisation vs deposit halo by segment", barmode="group",
                      yaxis_title="Amount over the horizon", height=400, margin=dict(t=50, b=40))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(seg.style.format({"users": "{:,}", "adoption_rate": "{:.1%}", "cannibalisation": "{:,.0f}",
                                   "deposit_halo": "{:,.0f}", "contribution": "{:,.0f}",
                                   "contribution_per_adopter": "{:,.2f}"}), width="stretch")

with tab4:
    st.subheader("Target vs control design")
    d1, d2, d3 = st.columns(3)
    d1.metric("Baseline adoption (no cashback)", f"{baseline:.1%}")
    d2.metric("Users needed per group", f"{needed:,}")
    d3.metric("Share of base in the test", f"{2 * needed / len(users):.1%}")
    st.write(f"Design: 5% significance, 80% power, detecting a lift of {mde * 100:.1f} points over control.")
    if 2 * needed <= len(users):
        sim = ex.simulate_test(users, cashback, 0.0, needed)
        st.subheader("Simulated read-out")
        r1, r2, r3 = st.columns(3)
        r1.metric("Target adoption", f"{sim['target_rate']:.1%}")
        r2.metric("Control adoption", f"{sim['control_rate']:.1%}")
        r3.metric("Lift (points)", f"{sim['absolute_lift'] * 100:.1f}",
                  f"95% CI {sim['ci_low'] * 100:.1f} to {sim['ci_high'] * 100:.1f}", delta_color="off")
        st.write(f"p-value {sim['p_value']:.4f}. "
                 f"{'Significant' if sim['significant'] else 'Not significant'} at 5%. "
                 f"Sample ratio check: {'FAILED' if sim['srm']['mismatch'] else 'passed'}.")
    else:
        st.warning("The simulated base is too small for this test. Raise the detectable lift or the base size.")

with tab5:
    st.markdown(write_memo(pnl, needed))
