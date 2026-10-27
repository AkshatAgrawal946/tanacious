"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import ml_model
from src.compare import MODELS, compare_models
from src.data import load_data
from src.generate_data import generate
from src.reconstruct import reconstruct_building_month

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
COLORS = {"actual": "#3b3b3b", "Baseline": "#1f77b4", "Random forest": "#ff7f0e"}
ACTUAL_LABEL = {"hidden ground truth": "True load (hidden, demo only)", "held-out meter": "Real meter reading"}

st.set_page_config(page_title="The Missing Meter", page_icon="⚡", layout="wide")


@st.cache_data
def campus_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Campus meter and bill data, generated on first launch if it is not on disk yet."""
    if not (DATA_DIR / "interval_15min.csv").exists() or not (DATA_DIR / "monthly_totals.csv").exists():
        generate(DATA_DIR)
    return load_data(DATA_DIR)


@st.cache_resource
def forest(_interval: pd.DataFrame, _bills: pd.DataFrame):
    """The saved Random Forest from ``python -m src.pipeline``, or a fresh one if none is saved."""
    if ml_model.MODEL_PATH.exists():
        return ml_model.load()
    return ml_model.train(_interval, _bills)


@st.cache_data
def comparison(_interval: pd.DataFrame, _bills: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both models' predictions and scores, from the pipeline outputs or computed now."""
    series_path, metrics_path = PROCESSED_DIR / "model_comparison_15min.csv", PROCESSED_DIR / "model_comparison_metrics.csv"
    if series_path.exists() and metrics_path.exists():
        return pd.read_csv(series_path, parse_dates=["timestamp"]), pd.read_csv(metrics_path)
    if not (DATA_DIR / "ground_truth_15min.csv").exists():
        generate(DATA_DIR)
    hidden = pd.read_csv(DATA_DIR / "ground_truth_15min.csv", parse_dates=["timestamp"])
    return compare_models(_interval, _bills, hidden)


st.title("⚡ The Missing Meter")
st.caption("Bill-constrained 15-minute demand reconstruction for campus facilities teams.")

interval, bills = campus_data()

building = st.sidebar.selectbox("Building", sorted(bills["building"].unique()))
months = sorted(bills.loc[bills["building"] == building, "month"].astype(str).unique())
month = st.sidebar.selectbox("Billing month", months)
model_name = st.sidebar.radio("Reconstruction model", MODELS)
shape = ml_model.shape_fn(forest(interval, bills), interval) if model_name == "Random forest" else None
result = reconstruct_building_month(interval, bills, building, month, shape)
bill = bills[(bills["building"] == building) & (bills["month"] == month)].iloc[0]

reconstruction_tab, compare_tab = st.tabs(["Reconstruction", "Compare models"])

with reconstruction_tab:
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
    st.caption(f"Model: {model_name}. Data coverage: {int(result.is_observed.sum()):,} observed intervals and {int(result.is_reconstructed.sum()):,} reconstructed intervals.")

    fig = go.Figure()
    estimated = result[result["is_reconstructed"]]
    actual = result[result["is_observed"]]
    fig.add_trace(go.Scatter(x=estimated["timestamp"], y=estimated["kWh_consumed"], mode="lines", name="Reconstructed estimate", line={"color": COLORS[model_name]}))
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["timestamp"], y=actual["kWh_consumed"], mode="markers", name="Observed smart-meter reading", marker={"color": "#1b7f3a", "size": 4}))
    fig.update_layout(title=f"{building}: reconstructed 15-minute demand ({model_name})", xaxis_title="Time", yaxis_title="kWh per 15 minutes", height=430)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        profile = result.assign(slot=result["timestamp"].dt.hour + result["timestamp"].dt.minute / 60).groupby("slot", as_index=False)["kWh_consumed"].mean()
        profile_chart = px.line(profile, x="slot", y="kWh_consumed", title="Average intraday profile", color_discrete_sequence=[COLORS[model_name]])
        profile_chart.add_vrect(x0=16, x1=20, fillcolor="red", opacity=0.12, line_width=0, annotation_text="4-8 PM")
        profile_chart.update_xaxes(title="Hour of day")
        left.plotly_chart(profile_chart, width="stretch")
    with right:
        if model_name == "Random forest":
            how = [f"{bill.building_type} daily schedule is a model input", "weekend flag", "temperature, 3-hour mean and cooling x occupancy", "hard reconciliation constraint"]
        else:
            how = [f"{bill.building_type} daily schedule re-times the metered profile", "weekday/weekend 15-minute reference shape", "cooling carried in the metered reference profile", "hard reconciliation constraint"]
        drivers = pd.DataFrame({"Driver": ["Building type", "Calendar", "Temperature", "Monthly bill"], "How it is used": how})
        right.subheader("Why this curve looks this way")
        right.dataframe(drivers, hide_index=True, width="stretch")
        right.info("Reconstructed readings are estimates. They are not smart-meter observations and should be validated against a real sub-meter pilot.")

with compare_tab:
    series, metrics = comparison(interval, bills)
    picked = series[series["building"] == building].dropna(subset=["actual"])
    evaluated_against = picked["evaluated_against"].iloc[0]
    actual_label = ACTUAL_LABEL[evaluated_against]
    if evaluated_against == "held-out meter":
        st.caption(f"{building} has a smart meter. For a fair test both models rebuild it from the *other* meters and its bill only, then are checked against its real readings.")
    else:
        st.caption(f"{building} only has a monthly bill. Both models rebuild it from the metered buildings; the hidden true load exists only because the data is synthetic.")

    building_metrics = metrics[metrics["building"] == building].set_index("model")
    columns = st.columns(len(MODELS))
    for column, name in zip(columns, MODELS):
        row = building_metrics.loc[name]
        column.markdown(f"**{name}**")
        a, b, c = column.columns(3)
        a.metric("Error per interval (MAPE)", f"{row.mape_pct:.1f}%")
        b.metric("4-8 PM energy error", f"{row.peak_window_error_pct:.1f}%")
        c.metric("Weekday peak", f"{int(row.predicted_peak_hour)}:00", delta=f"true {int(row.true_peak_hour)}:00", delta_color="off")

    traces = [("actual", actual_label, "solid"), ("Baseline", "Baseline", "solid"), ("Random forest", "Random forest", "dash")]
    weekday = picked[picked["timestamp"].dt.day_of_week < 5]
    profile = weekday.groupby(weekday["timestamp"].dt.hour + weekday["timestamp"].dt.minute / 60)[["actual", *MODELS]].mean()
    profile_fig = go.Figure()
    for column, label, dash in traces:
        profile_fig.add_trace(go.Scatter(x=profile.index, y=profile[column], mode="lines", name=label,
                                         line={"color": COLORS[column], "dash": dash, "width": 3 if column == "actual" else 2}))
    profile_fig.add_vrect(x0=16, x1=20, fillcolor="red", opacity=0.08, line_width=0, annotation_text="4-8 PM")
    profile_fig.update_layout(title=f"{building}: average weekday profile, all months", xaxis_title="Hour of day",
                              yaxis_title="kWh per 15 minutes", height=420, legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(profile_fig, width="stretch")

    month_rows = picked[picked["timestamp"].dt.to_period("M").astype(str) == month]
    first_monday = month_rows.loc[month_rows["timestamp"].dt.day_of_week == 0, "timestamp"].dt.normalize().min()
    week = month_rows[(month_rows["timestamp"] >= first_monday) & (month_rows["timestamp"] < first_monday + pd.Timedelta(days=7))]
    week_fig = go.Figure()
    for column, label, dash in traces:
        week_fig.add_trace(go.Scatter(x=week["timestamp"], y=week[column], mode="lines", name=label,
                                      line={"color": COLORS[column], "dash": dash, "width": 2 if column == "actual" else 1.5}))
    week_fig.update_layout(title=f"{building}: week of {first_monday:%d %b %Y} (Mon-Sun)", xaxis_title="Time",
                           yaxis_title="kWh per 15 minutes", height=420, legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(week_fig, width="stretch")

    st.subheader("All buildings")
    left, right = st.columns(2)
    for column, value, title in [(left, "mape_pct", "Error per interval (MAPE, %) - lower is better"),
                                 (right, "peak_window_error_pct", "4-8 PM energy error (%) - lower is better")]:
        bars = px.bar(metrics, x="building", y=value, color="model", barmode="group", title=title,
                      color_discrete_map=COLORS, labels={value: "%", "building": ""})
        column.plotly_chart(bars, width="stretch")
    table = metrics.assign(
        **{"Error per interval (MAPE)": metrics["mape_pct"].map("{:.1f}%".format),
           "4-8 PM energy error": metrics["peak_window_error_pct"].map("{:.1f}%".format),
           "Weekday peak (predicted / true)": metrics["predicted_peak_hour"].astype(int).astype(str) + ":00 / "
           + metrics["true_peak_hour"].astype(int).astype(str) + ":00"})
    st.dataframe(table[["building", "evaluated_against", "model", "Error per interval (MAPE)", "4-8 PM energy error",
                        "Weekday peak (predicted / true)"]], hide_index=True, width="stretch")
