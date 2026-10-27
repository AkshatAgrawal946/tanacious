"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src.data import load_data
from src.generate_data import generate
from src.reconstruct import reconstruct_building_month

DATA_DIR = Path("data/raw")

st.set_page_config(page_title="The Missing Meter", page_icon="⚡", layout="wide")


@st.cache_data
def campus_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Campus meter and bill data, generated on first launch if it is not on disk yet."""
    if not (DATA_DIR / "interval_15min.csv").exists() or not (DATA_DIR / "monthly_totals.csv").exists():
        generate(DATA_DIR)
    return load_data(DATA_DIR)


st.title("⚡ The Missing Meter")
st.caption("Bill-constrained 15-minute demand reconstruction for campus facilities teams.")

interval, bills = campus_data()

building = st.sidebar.selectbox("Building", sorted(bills["building"].unique()))
months = sorted(bills.loc[bills["building"] == building, "month"].astype(str).unique())
month = st.sidebar.selectbox("Billing month", months)
result = reconstruct_building_month(interval, bills, building, month)
bill = bills[(bills["building"] == building) & (bills["month"] == month)].iloc[0]

hour = result["timestamp"].dt.hour
peak_mask = (hour >= 16) & (hour < 20)
peak_kwh = float(result.loc[peak_mask, "kWh_consumed"].sum())
peak_share = peak_kwh / float(result["kWh_consumed"].sum())
peak_interval = result.loc[result["kWh_consumed"].idxmax(), "timestamp"].strftime("%d %b, %H:%M")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly bill", f"{float(bill.monthly_kWh):,.0f} kWh")
c2.metric("4-8 PM energy", f"{peak_kwh:,.0f} kWh")
c3.metric("4-8 PM share", f"{peak_share:.1%}")
c4.metric("Peak interval", peak_interval)

st.success(f"Accounting check passed: reconstructed sum = {result.kWh_consumed.sum():,.2f} kWh; bill = {float(bill.monthly_kWh):,.2f} kWh.")
st.caption(f"Data coverage: {int(result.is_observed.sum()):,} observed intervals and {int(result.is_reconstructed.sum()):,} reconstructed intervals.")

fig = go.Figure()
estimated = result[result["is_reconstructed"]]
actual = result[result["is_observed"]]
fig.add_trace(go.Scatter(x=estimated["timestamp"], y=estimated["kWh_consumed"], mode="lines", name="Reconstructed estimate", line={"color": "#1f77b4"}))
if not actual.empty:
    fig.add_trace(go.Scatter(x=actual["timestamp"], y=actual["kWh_consumed"], mode="markers", name="Observed smart-meter reading", marker={"color": "#1b7f3a", "size": 4}))
fig.update_layout(title=f"{building}: reconstructed 15-minute demand", xaxis_title="Time", yaxis_title="kWh per 15 minutes", height=430)
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    profile = result.assign(slot=result["timestamp"].dt.hour + result["timestamp"].dt.minute / 60).groupby("slot", as_index=False)["kWh_consumed"].mean()
    profile_chart = px.line(profile, x="slot", y="kWh_consumed", title="Average intraday profile")
    profile_chart.add_vrect(x0=16, x1=20, fillcolor="red", opacity=0.12, line_width=0, annotation_text="4-8 PM")
    profile_chart.update_xaxes(title="Hour of day")
    left.plotly_chart(profile_chart, width="stretch")
with right:
    drivers = pd.DataFrame({
        "Driver": ["Building type", "Calendar", "Temperature proxy", "Monthly bill"],
        "How it is used": [f"{bill.building_type} daily schedule re-times the metered profile", "weekday/weekend 15-minute reference shape", "cooling carried in the metered reference profile", "hard reconciliation constraint"],
    })
    right.subheader("Why this curve looks this way")
    right.dataframe(drivers, hide_index=True, width="stretch")
    right.info("Reconstructed readings are estimates. They are not smart-meter observations and should be validated against a real sub-meter pilot.")
