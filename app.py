"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import ml_model
from src.compare import MODELS, compare_models, score_series
from src.data import load_data
from src.generate_data import generate
from src.reconstruct import reconstruct_all

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
# Validated categorical palette (colour-blind safe, fixed order). Each building and each month
# keeps its slot, so changing the selection never repaints a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#52514e"
MODEL_COLORS = {"actual": NEUTRAL, "Baseline": SERIES[0], "Random forest": SERIES[1]}
ACTUAL_LABEL = {"hidden ground truth": "True load (hidden, demo only)", "held-out meter": "Real meter reading"}
DAY_TYPES = {
    "Weekdays": lambda ts: ts.dt.day_of_week < 5,
    "Weekends": lambda ts: ts.dt.day_of_week >= 5,
    "All days": lambda ts: pd.Series(True, index=ts.index),
}
DAY_LABELS = {"Weekdays": "weekday", "Weekends": "weekend day", "All days": "day"}
MAX_MONTHS = len(SERIES)

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
def reconstruction(model_name: str, _interval: pd.DataFrame, _bills: pd.DataFrame) -> pd.DataFrame:
    """Every building and month reconstructed by one model, with average power in kW."""
    shape = ml_model.shape_fn(forest(_interval, _bills), _interval) if model_name == "Random forest" else None
    result = reconstruct_all(_interval, _bills, shape)
    return result.assign(kW=result["kWh_consumed"] * 4)


@st.cache_data
def comparison(_interval: pd.DataFrame, _bills: pd.DataFrame) -> pd.DataFrame:
    """Both models' per-interval predictions, from the pipeline output or computed now."""
    series_path = PROCESSED_DIR / "model_comparison_15min.csv"
    if series_path.exists():
        series = pd.read_csv(series_path, parse_dates=["timestamp"])
    else:
        if not (DATA_DIR / "ground_truth_15min.csv").exists():
            generate(DATA_DIR)
        hidden = pd.read_csv(DATA_DIR / "ground_truth_15min.csv", parse_dates=["timestamp"])
        series, _ = compare_models(_interval, _bills, hidden)
    return series.assign(month=series["timestamp"].dt.to_period("M").astype(str))


@st.cache_data
def monthly_scores(series: pd.DataFrame) -> pd.DataFrame:
    """Scores for each single month; their maxima fix the error charts' axes."""
    return pd.concat([score_series(group).assign(month=month) for month, group in series.groupby("month")])


def month_label(month: str) -> str:
    return pd.Period(month, freq="M").strftime("%b %Y")


def hour_of_day(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.hour + timestamps.dt.minute / 60


def average_day(frame: pd.DataFrame, columns: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """Mean of ``columns`` per 15-minute slot of the day (and per ``by`` groups)."""
    keys = [*(by or []), hour_of_day(frame["timestamp"]).rename("hour")]
    return frame.groupby(keys)[columns].mean().reset_index()


def axis_top(values) -> float:
    return float(pd.Series(values).max()) * 1.08


def layout(fig: go.Figure, title: str, y_title: str, y_top: float, x_title: str = "Hour of day",
           height: int = 420, peak_band: bool = True) -> go.Figure:
    if peak_band:
        fig.add_vrect(x0=16, x1=20, fillcolor="#e34948", opacity=0.07, line_width=0,
                      annotation_text="4-8 PM", annotation_position="top left")
    fig.update_layout(title=title, height=height, hovermode="x unified", margin={"t": 60, "b": 40},
                      legend={"orientation": "h", "y": -0.18})
    fig.update_yaxes(title=y_title, range=[0, y_top], rangemode="tozero")
    if x_title == "Hour of day":
        fig.update_xaxes(title=x_title, range=[0, 24], tickvals=list(range(0, 25, 3)),
                         ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)])
    else:
        fig.update_xaxes(title=x_title)
    return fig


def monthly_bars(bills: pd.DataFrame, buildings: list[str], months: list[str], colors: dict[str, str],
                 title: str) -> go.Figure:
    """Whole-year monthly energy (from the bills), stacked by building; selected months stand out."""
    fig = go.Figure()
    all_months = sorted(bills["month"].unique())
    for building in buildings:
        energy = bills[bills["building"] == building].set_index("month")["monthly_kWh"].reindex(all_months) / 1000
        fig.add_trace(go.Bar(
            x=[month_label(m) for m in all_months], y=energy, name=building,
            marker={"color": colors[building], "opacity": [1.0 if m in months else 0.3 for m in all_months],
                    "line": {"width": 1, "color": "white"}},
            hovertemplate="%{x}: %{y:,.1f} MWh<extra>" + building + "</extra>"))
    fig.update_layout(barmode="stack", title=title, height=380, margin={"t": 60, "b": 40},
                      legend={"orientation": "h", "y": -0.2})
    fig.update_yaxes(title="Energy (MWh)")
    return fig


st.title("⚡ The Missing Meter")
st.caption("Bill-constrained 15-minute demand reconstruction for campus facilities teams.")

interval, bills = campus_data()
bills = bills.assign(month=bills["month"].astype(str))
buildings = sorted(bills["building"].unique())
building_colors = {b: SERIES[i % len(SERIES)] for i, b in enumerate(buildings)}
all_months = sorted(bills["month"].unique())
meter_status = {
    b: ("Smart meter" if interval.loc[interval["building"] == b, "kWh_consumed"].count() > 0.5 * 96 * 365
        else "Bill + pilot meter" if b in set(interval["building"]) else "Bill only")
    for b in buildings
}

with st.sidebar:
    building = st.selectbox("Building", buildings)
    months = st.multiselect("Months", all_months, default=all_months[:1], format_func=month_label,
                            max_selections=MAX_MONTHS,
                            help=f"Pick up to {MAX_MONTHS} months to compare them side by side.")
    day_type = st.radio("Average over", list(DAY_TYPES), horizontal=True)
    model_name = st.radio("Reconstruction model", MODELS)
if not months:
    st.info("Pick at least one month in the sidebar.")
    st.stop()
months = sorted(months)
month_colors = {m: SERIES[i] for i, m in enumerate(months)}
period = ", ".join(month_label(m) for m in months)

rec = reconstruction(model_name, interval, bills)
rec_days = rec[DAY_TYPES[day_type](rec["timestamp"])]
selected = rec[rec["month"].isin(months)]
selected_days = rec_days[rec_days["month"].isin(months)]

campus_tab, building_tab, compare_tab = st.tabs(["Campus overview", "Building", "Compare models"])

with campus_tab:
    campus = selected.groupby("timestamp")["kW"].sum()
    energy = bills[bills["month"].isin(months)].groupby("building")["monthly_kWh"].sum()
    peak_window = selected["timestamp"].dt.hour.between(16, 19)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campus energy", f"{energy.sum() / 1000:,.1f} MWh", help="Sum of the monthly bills for the selected months.")
    c2.metric("4-8 PM share", f"{selected.loc[peak_window, 'kWh_consumed'].sum() / selected['kWh_consumed'].sum():.1%}")
    c3.metric("Campus peak", f"{campus.max():,.0f} kW", delta=campus.idxmax().strftime("%d %b, %H:%M"), delta_color="off")
    c4.metric("Biggest user", energy.idxmax(), delta=f"{energy.max() / energy.sum():.0%} of campus energy", delta_color="off")

    profile = average_day(selected_days, ["kW"], ["building"])
    # Fix the axis to the busiest single month of the year so switching months keeps the scale.
    yearly = average_day(rec_days, ["kW"], ["month", "building"]).groupby(["month", "hour"])["kW"].sum()
    fig = go.Figure()
    for b in buildings:
        rows = profile[profile["building"] == b]
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["kW"], name=b, stackgroup="campus", mode="lines",
                                 line={"width": 0.5, "color": building_colors[b]}, fillcolor=building_colors[b],
                                 hovertemplate="%{y:,.0f} kW<extra>" + b + "</extra>"))
    st.plotly_chart(layout(fig, f"Campus load on an average {DAY_LABELS[day_type]} - {period}",
                           "Average power (kW)", axis_top(yearly)), width="stretch")
    st.caption("Each band is one building; the top edge is the whole campus. "
               "The y-axis is fixed to the busiest month of the year, so months can be compared by eye.")

    st.plotly_chart(monthly_bars(bills, buildings, months, building_colors,
                                 "Monthly energy by building (selected months highlighted)"), width="stretch")

    day_profile = profile.set_index(["building", "hour"])["kW"]
    summary = pd.DataFrame({
        "Metering": pd.Series(meter_status),
        "Energy (MWh)": (energy / 1000).round(1),
        "Share of campus": (energy / energy.sum()).map("{:.0%}".format),
        "4-8 PM share": (selected[peak_window].groupby("building")["kWh_consumed"].sum()
                         / selected.groupby("building")["kWh_consumed"].sum()).map("{:.0%}".format),
        "Average-day peak (kW)": day_profile.groupby("building").max().round(0),
        "Peak time": day_profile.groupby("building").idxmax().map(lambda key: f"{int(key[1]):02d}:{int(key[1] % 1 * 60):02d}"),
    }).rename_axis("Building")
    st.dataframe(summary, width="stretch")

with building_tab:
    own = selected[selected["building"] == building]
    own_bills = bills[(bills["building"] == building) & bills["month"].isin(months)]
    peak_row = own.loc[own["kW"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Billed energy", f"{own_bills['monthly_kWh'].sum():,.0f} kWh")
    c2.metric("4-8 PM share", f"{own.loc[own['timestamp'].dt.hour.between(16, 19), 'kWh_consumed'].sum() / own['kWh_consumed'].sum():.1%}")
    c3.metric("Highest 15-min load", f"{peak_row.kW:,.0f} kW", delta=peak_row.timestamp.strftime("%d %b, %H:%M"), delta_color="off")
    c4.metric("Real readings", f"{own['is_observed'].mean():.0%}", delta=meter_status[building], delta_color="off")
    st.success(f"Accounting check passed: reconstructed total = {own['kWh_consumed'].sum():,.1f} kWh; "
               f"billed = {own_bills['monthly_kWh'].sum():,.1f} kWh ({model_name}).")

    own_year = rec_days[rec_days["building"] == building]
    by_month = average_day(own_year, ["kW"], ["month"])
    fig = go.Figure()
    for m in months:
        rows = by_month[by_month["month"] == m]
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["kW"], name=month_label(m), mode="lines",
                                 line={"width": 2, "color": month_colors[m]},
                                 hovertemplate="%{y:,.1f} kW<extra>" + month_label(m) + "</extra>"))
        if len(months) <= 4:  # direct labels while they stay readable; the legend covers the rest
            fig.add_annotation(x=rows["hour"].iloc[-1], y=rows["kW"].iloc[-1], text=pd.Period(m, "M").strftime("%b"),
                               showarrow=False, xanchor="left", xshift=4, font={"color": NEUTRAL})
    st.plotly_chart(layout(fig, f"{building}: average {DAY_LABELS[day_type]} by month",
                           "Average power (kW)", axis_top(by_month["kW"])), width="stretch")

    fig = go.Figure()
    for m in months:
        rows = own[own["month"] == m]
        fig.add_trace(go.Scattergl(x=rows["timestamp"], y=rows["kW"], name=month_label(m), mode="lines",
                                   line={"width": 1, "color": month_colors[m]},
                                   hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kW<extra></extra>"))
    year_own = rec[rec["building"] == building]
    st.plotly_chart(layout(fig, f"{building}: 15-minute load ({model_name})", "Power (kW)",
                           axis_top(year_own["kW"]), x_title="Date", peak_band=False), width="stretch")
    st.caption(f"{int(own['is_observed'].sum()):,} of {len(own):,} intervals are real meter readings; the rest are "
               "reconstructed and scaled so each month matches its bill.")

    st.plotly_chart(monthly_bars(bills, [building], months, building_colors,
                                 f"{building}: monthly energy (selected months highlighted)"), width="stretch")
    with st.expander("Why this curve looks this way"):
        building_type = bills.loc[bills["building"] == building, "building_type"].iloc[0]
        if model_name == "Random forest":
            how = [f"{building_type} daily schedule is a model input", "weekend and break flags", "temperature, 3-hour mean and cooling x occupancy", "hard reconciliation constraint"]
        else:
            how = [f"{building_type} daily schedule re-times the metered profile", "separate term and break weekly profiles", "cooling carried in the metered reference profile", "hard reconciliation constraint"]
        st.dataframe(pd.DataFrame({"Driver": ["Building type", "Calendar", "Temperature", "Monthly bill"], "How it is used": how}),
                     hide_index=True, width="stretch")
        st.info("Reconstructed readings are estimates, not smart-meter observations.")

with compare_tab:
    series = comparison(interval, bills)
    per_month = monthly_scores(series)
    scored = series[~series["is_pilot"]].dropna(subset=["actual"])
    in_period = scored[scored["month"].isin(months)]
    metrics = score_series(in_period)
    picked = in_period[in_period["building"] == building]
    evaluated_against = picked["evaluated_against"].iloc[0]
    if evaluated_against == "held-out meter":
        st.caption(f"{building} has a smart meter. For a fair test both models rebuild it from the *other* meters and "
                   "its bill only, then are checked against its real readings.")
    else:
        st.caption(f"{building} only has a monthly bill. Both models rebuild it from the metered buildings and pilot-meter "
                   "weeks; the hidden true load exists only because the data is synthetic. Pilot weeks are not scored.")

    building_metrics = metrics[metrics["building"] == building].set_index("model")
    for column, name in zip(st.columns(len(MODELS)), MODELS):
        row = building_metrics.loc[name]
        column.markdown(f"**{name}** - {period}")
        a, b, c = column.columns(3)
        a.metric("Error per interval (MAPE)", f"{row.mape_pct:.1f}%")
        b.metric("4-8 PM energy error", f"{row.peak_window_error_pct:.1f}%")
        c.metric("Weekday peak", f"{int(row.predicted_peak_hour)}:00", delta=f"true {int(row.true_peak_hour)}:00", delta_color="off")

    lines = [("actual", ACTUAL_LABEL[evaluated_against], "solid", 3), ("Baseline", "Baseline", "solid", 2),
             ("Random forest", "Random forest", "dash", 2)]
    days = DAY_TYPES[day_type]
    own_scored = scored[(scored["building"] == building) & days(scored["timestamp"])]
    profile = average_day(own_scored[own_scored["month"].isin(months)], ["actual", *MODELS])
    yearly = average_day(own_scored, ["actual", *MODELS], ["month"])
    fig = go.Figure()
    for column, label, dash, width in lines:
        fig.add_trace(go.Scatter(x=profile["hour"], y=profile[column] * 4, mode="lines", name=label,
                                 line={"color": MODEL_COLORS[column], "dash": dash, "width": width},
                                 hovertemplate="%{y:,.1f} kW<extra>" + label + "</extra>"))
    st.plotly_chart(layout(fig, f"{building}: average {DAY_LABELS[day_type]}, {period}", "Average power (kW)",
                           axis_top(yearly[["actual", *MODELS]].max(axis=1) * 4)), width="stretch")

    first = scored[(scored["building"] == building) & (scored["month"] == months[0])]
    monday = first.loc[first["timestamp"].dt.day_of_week == 0, "timestamp"].dt.normalize().min()
    week = first[(first["timestamp"] >= monday) & (first["timestamp"] < monday + pd.Timedelta(days=7))]
    fig = go.Figure()
    for column, label, dash, width in lines:
        fig.add_trace(go.Scatter(x=week["timestamp"], y=week[column] * 4, mode="lines", name=label,
                                 line={"color": MODEL_COLORS[column], "dash": dash, "width": max(width - 1, 1)},
                                 hovertemplate="%{y:,.1f} kW<extra>" + label + "</extra>"))
    own_all = scored[scored["building"] == building]
    st.plotly_chart(layout(fig, f"{building}: week of {monday:%d %b %Y} (Mon-Sun)", "Power (kW)",
                           axis_top(own_all[["actual", *MODELS]].max(axis=1) * 4), x_title="Date", peak_band=False),
                    width="stretch")

    st.subheader(f"All buildings - {period}")
    left, right = st.columns(2)
    for column, value, title in [(left, "mape_pct", "Error per interval (MAPE) - lower is better"),
                                 (right, "peak_window_error_pct", "4-8 PM energy error - lower is better")]:
        fig = go.Figure()
        for name in MODELS:
            rows = metrics[metrics["model"] == name]
            fig.add_trace(go.Bar(x=rows["building"], y=rows[value], name=name,
                                 marker={"color": MODEL_COLORS[name], "line": {"width": 1, "color": "white"}},
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>"))
        fig.update_layout(barmode="group", title=title, height=360, margin={"t": 60, "b": 40},
                          legend={"orientation": "h", "y": -0.25})
        fig.update_yaxes(title="%", range=[0, axis_top(per_month[value])])
        column.plotly_chart(fig, width="stretch")
    table = metrics.assign(**{
        "Error per interval (MAPE)": metrics["mape_pct"].map("{:.1f}%".format),
        "4-8 PM energy error": metrics["peak_window_error_pct"].map("{:.1f}%".format),
        "Weekday peak (predicted / true)": metrics["predicted_peak_hour"].astype(int).astype(str) + ":00 / "
        + metrics["true_peak_hour"].astype(int).astype(str) + ":00"})
    st.dataframe(table[["building", "evaluated_against", "model", "Error per interval (MAPE)", "4-8 PM energy error",
                        "Weekday peak (predicted / true)"]], hide_index=True, width="stretch")
