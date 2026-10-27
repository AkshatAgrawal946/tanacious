"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import ml_model
from src.data import load_data
from src.generate_data import generate
from src.reconstruct import reconstruct_all

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
# Validated categorical palette (colour-blind safe, fixed order). Each building and each month
# keeps its slot, so changing the selection never repaints a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PEAK = "#e34948"
MAX_MONTHS = len(SERIES)

st.set_page_config(page_title="The Missing Meter", page_icon=":material/bolt:", layout="wide")


@st.cache_data
def campus_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Campus meter and bill data, generated on first launch if it is not on disk yet."""
    if not (DATA_DIR / "interval_15min.csv").exists() or not (DATA_DIR / "monthly_totals.csv").exists():
        generate(DATA_DIR)
    return load_data(DATA_DIR)


@st.cache_resource
def forest(_interval: pd.DataFrame, _bills: pd.DataFrame):
    """The saved Random Forest from ``python -m src.pipeline``, or a fresh one if none is saved
    (or the saved one was trained with an older set of inputs)."""
    if ml_model.MODEL_PATH.exists():
        model = ml_model.load()
        if list(model.feature_names_in_) == ml_model.FEATURES:
            return model
    return ml_model.train(_interval, _bills)


@st.cache_data
def reconstruction(_interval: pd.DataFrame, _bills: pd.DataFrame) -> pd.DataFrame:
    """Every building and month reconstructed by the Random Forest, with average power in kW."""
    result = reconstruct_all(_interval, _bills, ml_model.shape_fn(forest(_interval, _bills), _interval, _bills))
    return result.assign(kW=result["kWh_consumed"] * 4)


@st.cache_data
def tested_accuracy() -> pd.DataFrame | None:
    """Per-building accuracy from the last ``python -m src.pipeline`` run, if available."""
    path = PROCESSED_DIR / "evaluation_metrics.csv"
    return pd.read_csv(path).set_index("building") if path.exists() else None


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


def pretty(building: str) -> str:
    return building.replace("_", " ")


def layout(fig: go.Figure, y_title: str, y_top: float, x_title: str = "Hour of day",
           height: int = 380, peak_band: bool = True, right_margin: int = 10) -> go.Figure:
    if peak_band:
        fig.add_vrect(x0=16, x1=20, fillcolor=PEAK, opacity=0.08, line_width=0,
                      annotation_text="4-8 PM peak", annotation_position="top left",
                      annotation_font={"size": 11, "color": PEAK})
    fig.update_layout(height=height, hovermode="x unified", margin={"t": 24, "b": 10, "l": 10, "r": right_margin},
                      legend={"orientation": "h", "y": -0.2, "x": 0, "title": None},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title=y_title, range=[0, y_top], rangemode="tozero", zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    if x_title == "Hour of day":
        fig.update_xaxes(title=x_title, range=[0, 24], tickvals=list(range(0, 25, 3)),
                         ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)])
    else:
        fig.update_xaxes(title=x_title)
    return fig


def chart_card(title: str, fig: go.Figure, caption: str | None = None) -> None:
    """A chart in a bordered card with its heading and an optional note underneath."""
    with st.container(border=True):
        st.markdown(f"##### {title}")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        if caption:
            st.caption(caption)


def monthly_bars(bills: pd.DataFrame, buildings: list[str], months: list[str], colors: dict[str, str]) -> go.Figure:
    """Whole-year monthly energy (from the bills), stacked by building; selected months stand out."""
    fig = go.Figure()
    all_months = sorted(bills["month"].unique())
    for building in buildings:
        energy = bills[bills["building"] == building].set_index("month")["monthly_kWh"].reindex(all_months) / 1000
        fig.add_trace(go.Bar(
            x=[month_label(m) for m in all_months], y=energy, name=pretty(building),
            marker={"color": colors[building], "opacity": [1.0 if m in months else 0.28 for m in all_months],
                    "line": {"width": 0}},
            hovertemplate="%{x}: %{y:,.1f} MWh<extra>" + pretty(building) + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0.3, height=340, margin={"t": 10, "b": 10, "l": 10, "r": 10},
                      legend={"orientation": "h", "y": -0.18, "x": 0},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="Energy (MWh)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


interval, bills = campus_data()
bills = bills.assign(month=bills["month"].astype(str))
buildings = sorted(bills["building"].unique())
building_colors = {b: SERIES[i % len(SERIES)] for i, b in enumerate(buildings)}
all_months = sorted(bills["month"].unique())
meter_status = {
    b: "Smart meter" if interval.loc[interval["building"] == b, "kWh_consumed"].count() > 0.5 * 96 * 365 else "Bill only"
    for b in buildings
}

with st.sidebar:
    st.markdown("### :material/bolt: The Missing Meter")
    st.caption("15-minute demand, rebuilt from monthly bills.")
    months = st.multiselect("Months", all_months, default=all_months[:1], format_func=month_label,
                            max_selections=MAX_MONTHS, placeholder="Choose months",
                            help=f"Pick up to {MAX_MONTHS} months to compare them side by side.")
    st.divider()
    st.markdown("**Metering**")
    for b in buildings:
        smart = meter_status[b] == "Smart meter"
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(pretty(b), width="stretch")
            st.badge(meter_status[b], color="green" if smart else "gray",
                     icon=":material/sensors:" if smart else ":material/receipt_long:")

st.title("The Missing Meter")
st.caption("Bill-constrained 15-minute demand reconstruction for campus facilities teams.")

if not months:
    st.info("Pick at least one month in the sidebar to get started.", icon=":material/calendar_month:")
    st.stop()
months = sorted(months)
month_colors = {m: SERIES[i] for i, m in enumerate(months)}
period = ", ".join(month_label(m) for m in months)

rec = reconstruction(interval, bills)
selected = rec[rec["month"].isin(months)]

campus_tab, building_tab = st.tabs([":material/apartment: Campus overview", ":material/domain: Building"])

with campus_tab:
    campus = selected.groupby("timestamp")["kW"].sum()
    energy = bills[bills["month"].isin(months)].groupby("building")["monthly_kWh"].sum()
    peak_window = selected["timestamp"].dt.hour.between(16, 19)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campus energy", f"{energy.sum() / 1000:,.1f} MWh", border=True, height="stretch",
              help="Sum of the monthly bills for the selected months.")
    c2.metric("4-8 PM share", f"{selected.loc[peak_window, 'kWh_consumed'].sum() / selected['kWh_consumed'].sum():.1%}",
              border=True, height="stretch")
    c3.metric("Campus peak", f"{campus.max():,.0f} kW", delta=campus.idxmax().strftime("%d %b, %H:%M"),
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    c4.metric("Biggest user", pretty(energy.idxmax()), delta=f"{energy.max() / energy.sum():.0%} of campus energy",
              delta_color="off", delta_arrow="off", border=True, height="stretch")

    profile = average_day(selected, ["kW"], ["building"])
    # Fix the axis to the busiest single month of the year so switching months keeps the scale.
    yearly = average_day(rec, ["kW"], ["month", "building"]).groupby(["month", "hour"])["kW"].sum()
    fig = go.Figure()
    for b in buildings:
        rows = profile[profile["building"] == b]
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["kW"], name=pretty(b), stackgroup="campus", mode="lines",
                                 line={"width": 0.5, "color": building_colors[b]}, fillcolor=building_colors[b],
                                 hovertemplate="%{y:,.0f} kW<extra>" + pretty(b) + "</extra>"))
    chart_card(f"Campus load on an average day · {period}",
               layout(fig, "Average power (kW)", axis_top(yearly)),
               "Each band is one building; the top edge is the whole campus. "
               "The y-axis is fixed to the busiest month of the year, so months can be compared by eye.")

    chart_card("Monthly energy by building", monthly_bars(bills, buildings, months, building_colors),
               "Selected months are shown in full colour.")

    day_profile = profile.set_index(["building", "hour"])["kW"]
    summary = pd.DataFrame({
        "Metering": pd.Series(meter_status),
        "Energy": energy / 1000,
        "Share of campus": energy / energy.sum() * 100,
        "4-8 PM share": (selected[peak_window].groupby("building")["kWh_consumed"].sum()
                         / selected.groupby("building")["kWh_consumed"].sum() * 100),
        "Average-day peak": day_profile.groupby("building").max(),
        "Peak time": day_profile.groupby("building").idxmax().map(lambda key: f"{int(key[1]):02d}:{int(key[1] % 1 * 60):02d}"),
    })
    summary.index = summary.index.map(pretty)
    with st.container(border=True):
        st.markdown("##### Buildings at a glance")
        st.dataframe(summary.rename_axis("Building").reset_index(), width="stretch", hide_index=True,
                     column_config={
                         "Energy": st.column_config.NumberColumn(format="%.1f MWh"),
                         "Share of campus": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                         "4-8 PM share": st.column_config.NumberColumn(format="%.0f%%"),
                         "Average-day peak": st.column_config.NumberColumn(format="%.0f kW"),
                     })

with building_tab:
    building = st.segmented_control("Building", buildings, default=buildings[0], required=True,
                                    format_func=pretty, label_visibility="collapsed")
    own = selected[selected["building"] == building]
    own_bills = bills[(bills["building"] == building) & bills["month"].isin(months)]
    peak_row = own.loc[own["kW"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Billed energy", f"{own_bills['monthly_kWh'].sum():,.0f} kWh", border=True, height="stretch")
    c2.metric("4-8 PM share", f"{own.loc[own['timestamp'].dt.hour.between(16, 19), 'kWh_consumed'].sum() / own['kWh_consumed'].sum():.1%}",
              border=True, height="stretch")
    c3.metric("Highest 15-min load", f"{peak_row.kW:,.0f} kW", delta=peak_row.timestamp.strftime("%d %b, %H:%M"),
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    c4.metric("Real readings", f"{own['is_observed'].mean():.0%}", delta=meter_status[building],
              delta_color="off", delta_arrow="off", border=True, height="stretch")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge("Matches the bill", icon=":material/check_circle:", color="green",
                 help=f"Reconstructed total = {own['kWh_consumed'].sum():,.1f} kWh; "
                      f"billed = {own_bills['monthly_kWh'].sum():,.1f} kWh.")
        accuracy = tested_accuracy()
        if accuracy is not None and building in accuracy.index and own["is_reconstructed"].any():
            row = accuracy.loc[building]
            how = ("against the hidden true load" if row.evaluated_against == "hidden ground truth"
                   else "by rebuilding it with its own meter hidden")
            st.badge(f"{row.mape_pct:.1f}% avg. error per interval", icon=":material/target:", color="blue",
                     help=f"Tested over the year {how}.")
            st.badge(f"{row.peak_window_error_pct:.1f}% error on 4-8 PM energy", color="blue")

    own_year = rec[rec["building"] == building]
    by_month = average_day(own_year, ["kW"], ["month"])
    direct_labels = len(months) <= 4  # direct labels while they stay readable; the legend covers the rest
    fig = go.Figure()
    for m in months:
        rows = by_month[by_month["month"] == m]
        fig.add_trace(go.Scatter(x=rows["hour"], y=rows["kW"], name=month_label(m), mode="lines",
                                 line={"width": 2.5, "color": month_colors[m]},
                                 hovertemplate="%{y:,.1f} kW<extra>" + month_label(m) + "</extra>"))
        if direct_labels:
            fig.add_annotation(x=rows["hour"].iloc[-1], y=rows["kW"].iloc[-1], text=pd.Period(m, "M").strftime("%b"),
                               showarrow=False, xanchor="left", xshift=6, font={"color": month_colors[m], "size": 12})
    if direct_labels:
        fig.update_layout(showlegend=False)
    chart_card(f"{pretty(building)} · average day by month",
               layout(fig, "Average power (kW)", axis_top(by_month["kW"]), right_margin=44 if direct_labels else 10))

    fig = go.Figure()
    for m in months:
        rows = own[own["month"] == m]
        fig.add_trace(go.Scattergl(x=rows["timestamp"], y=rows["kW"], name=month_label(m), mode="lines",
                                   line={"width": 1, "color": month_colors[m]},
                                   hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kW<extra></extra>"))
    year_own = rec[rec["building"] == building]
    chart_card(f"{pretty(building)} · 15-minute load",
               layout(fig, "Power (kW)", axis_top(year_own["kW"]), x_title="Date", peak_band=False),
               f"{int(own['is_observed'].sum()):,} of {len(own):,} intervals are real meter readings; the rest are "
               "reconstructed and scaled so each month matches its bill.")

    chart_card(f"{pretty(building)} · monthly energy", monthly_bars(bills, [building], months, building_colors),
               "Selected months are shown in full colour.")
