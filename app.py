"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import ml_model, provenance
from src.data import load_data
from src.forecasting import registry, selection
from src.forecasting.backtest import fit_random_forest
from src.forecasting.config import ForecastExperimentConfig
from src.forecasting.data_prep import actuals, building_series, make_request
from src.generate_data import generate
from src.reconstruct import reconstruct_all

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EVALUATION_META = PROCESSED_DIR / "evaluation_metrics.meta.json"
FORECAST_CONFIG = Path("configs/forecast_benchmark.json")
# Validated categorical palette (colour-blind safe, fixed order). Each building and each month
# keeps its slot, so changing the selection never repaints a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PEAK = "#e34948"
ACTUAL, RECONSTRUCTED, FORECAST = "#2a78d6", "#8a8f9c", "#eb6834"
MAX_MONTHS = len(SERIES)

st.set_page_config(page_title="The Missing Meter", page_icon=":material/bolt:", layout="wide")


# Every cached function is keyed by content fingerprints (and the model version), never by
# underscore-prefixed arguments, which Streamlit leaves out of the cache key and would let a
# model or reconstruction built from older data be served after the data changed.
def data_files_signature() -> str:
    """Content hash of the input CSVs (generated on first launch if they are not on disk yet)."""
    if not (DATA_DIR / "interval_15min.csv").exists() or not (DATA_DIR / "monthly_totals.csv").exists():
        generate(DATA_DIR)
    return provenance.file_fingerprint(DATA_DIR / "interval_15min.csv", DATA_DIR / "monthly_totals.csv")


@st.cache_data
def campus_data(files_signature: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Campus meter and bill data plus their dataset fingerprint."""
    interval, bills = load_data(DATA_DIR)
    return interval, bills, provenance.dataset_fingerprint(interval, bills)


@st.cache_resource
def forest(files_signature: str, dataset_fingerprint: str, model_version: str = ml_model.MODEL_VERSION):
    """The saved Random Forest if its metadata says it was trained on exactly this dataset with this
    model version; otherwise a freshly trained one, saved with its metadata for the next launch."""
    model = ml_model.load_compatible(dataset_fingerprint)
    if model is None:
        interval, bills, _ = campus_data(files_signature)
        model = ml_model.train(interval, bills)
        ml_model.save(model, dataset_fingerprint=dataset_fingerprint)
    return model


@st.cache_data
def reconstruction(files_signature: str, dataset_fingerprint: str, model_version: str = ml_model.MODEL_VERSION) -> pd.DataFrame:
    """Every building and month reconstructed by the Random Forest, with average power in kW."""
    interval, bills, _ = campus_data(files_signature)
    model = forest(files_signature, dataset_fingerprint, model_version)
    result = reconstruct_all(interval, bills, ml_model.shape_fn(model, interval, bills))
    return result.assign(kW=result["kWh_consumed"] * 4)


@st.cache_data
def tested_accuracy(dataset_fingerprint: str, metrics_signature: str) -> pd.DataFrame | None:
    """Per-building accuracy from the last ``python -m src.pipeline`` run, only if that run scored
    this exact dataset (its metadata records the fingerprint)."""
    meta = provenance.read_json(EVALUATION_META)
    path = PROCESSED_DIR / "evaluation_metrics.csv"
    if not path.exists() or meta is None or meta.get("dataset_fingerprint") != dataset_fingerprint:
        return None
    return pd.read_csv(path).set_index("building")


@st.cache_resource
def forecast_forest(files_signature: str):
    interval, _, _ = campus_data(files_signature)
    return fit_random_forest(forecast_config(), interval)


@st.cache_resource
def forecaster_cache(files_signature: str) -> dict:
    """Loaded forecasters (Chronos-2 weights load once per session)."""
    return {}


def forecast_config() -> ForecastExperimentConfig:
    return ForecastExperimentConfig.from_json(FORECAST_CONFIG) if FORECAST_CONFIG.exists() else ForecastExperimentConfig()


@st.cache_data(show_spinner="Forecasting the next 24 hours...")
def day_ahead(files_signature: str, building: str, origin: str, model_name: str | None, summary_signature: str) -> dict:
    interval, _, _ = campus_data(files_signature)
    config = forecast_config()
    series = building_series(interval, building)
    request = make_request(series, building, pd.Timestamp(origin), config.context_steps, config.horizon_steps)
    chosen = selection.forecast_with_fallback(
        [request], model_name, selection.benchmark_summary(), device=config.device,
        train_forest=lambda: forecast_forest(files_signature), cache=forecaster_cache(files_signature))
    result = chosen.results[0]
    return {"frame": result.to_frame().assign(actual=actuals(series, request.origin, request.horizon).to_numpy()),
            "used": chosen.used, "requested": chosen.requested, "fallback_reason": chosen.fallback_reason,
            "version": result.model_version}


def month_label(month: str) -> str:
    return pd.Period(month, freq="M").strftime("%b %Y")


def hour_of_day(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.hour + timestamps.dt.minute / 60


def clock(hours):
    """Hour-of-day floats as times on a fixed date, so charts show 06:15 rather than 6.25."""
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(hours, unit="h")


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
        fig.add_vrect(x0=clock(16), x1=clock(20), fillcolor=PEAK, opacity=0.08, line_width=0,
                      annotation_text="4-8 PM peak", annotation_position="top left",
                      annotation_font={"size": 11, "color": PEAK})
    fig.update_layout(height=height, hovermode="x unified", margin={"t": 24, "b": 10, "l": 10, "r": right_margin},
                      legend={"orientation": "h", "y": -0.2, "x": 0, "title": None},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title=y_title, range=[0, y_top], rangemode="tozero", zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    if x_title == "Hour of day":
        fig.update_xaxes(title=x_title, range=[clock(0), clock(24)], hoverformat="%H:%M",
                         tickvals=[clock(h) for h in range(0, 25, 3)],
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


files_signature = data_files_signature()
interval, bills, fingerprint = campus_data(files_signature)
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

rec = reconstruction(files_signature, fingerprint)
selected = rec[rec["month"].isin(months)]

campus_tab, building_tab, forecast_tab = st.tabs([":material/apartment: Campus overview", ":material/domain: Building",
                                                  ":material/trending_up: Forecast"])

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
        fig.add_trace(go.Scatter(x=clock(rows["hour"]), y=rows["kW"], name=pretty(b), stackgroup="campus", mode="lines",
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
        accuracy = tested_accuracy(fingerprint, provenance.file_fingerprint(PROCESSED_DIR / "evaluation_metrics.csv", EVALUATION_META))
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
        fig.add_trace(go.Scatter(x=clock(rows["hour"]), y=rows["kW"], name=month_label(m), mode="lines",
                                 line={"width": 2.5, "color": month_colors[m]},
                                 hovertemplate="%{y:,.1f} kW<extra>" + month_label(m) + "</extra>"))
        if direct_labels:
            fig.add_annotation(x=clock(rows["hour"].iloc[-1]), y=rows["kW"].iloc[-1], text=pd.Period(m, "M").strftime("%b"),
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

with forecast_tab:
    config = forecast_config()
    summary = selection.benchmark_summary()
    summary_ok = summary is not None and summary.get("dataset_fingerprint") == fingerprint
    if summary is None:
        st.info("No forecast benchmark yet: run `python -m src.forecasting.backtest`. Until then the app uses the "
                "previous-week baseline by default and a 14-day history rule.", icon=":material/info:")
    elif not summary_ok:
        st.warning("The forecast benchmark was run on a different version of the data; its scores and default model "
                   "are shown for reference only. Re-run `python -m src.forecasting.backtest`.", icon=":material/warning:")
    measured = interval.groupby("building")["kWh_consumed"].count()
    candidates = sorted(measured[measured > 0].index, key=lambda b: (-measured[b], b))  # longest meter history first
    f1, f2, f3 = st.columns([2, 2, 3])
    building_f = f1.selectbox("Building", candidates, format_func=pretty, key="forecast_building")
    series = building_series(interval, building_f)
    first_origin = (series.index[0] + pd.Timedelta(days=14)).normalize() + pd.Timedelta(days=1)
    last_origin = (series.index[-1] - pd.Timedelta(days=1)).normalize() + pd.Timedelta(days=1)
    if first_origin > last_origin:
        first_origin = last_origin
    default_origin = min(max(pd.Timestamp(config.test[0]), first_origin), last_origin)
    origin_day = f2.date_input("Forecast the day starting", default_origin.date(), min_value=first_origin.date(),
                               max_value=last_origin.date(), key="forecast_origin")
    default_model = (summary or {}).get("default_model", selection.FALLBACK_MODEL)
    options = [None, "chronos2_zero_shot", "seasonal_previous_week", "seasonal_previous_day", "random_forest_forecast",
               *[n for n, r in registry.load().items() if r.kind == "local_checkpoint" and r.status == "verified"]]
    model_choice = f3.selectbox("Model", options, key="forecast_model",
                                format_func=lambda m: f"Automatic ({default_model})" if m is None else m,
                                help="Automatic uses the model that passed the benchmark's promotion gate, "
                                     "otherwise the strongest simple baseline.")
    origin = pd.Timestamp(origin_day)
    routing = selection.route(series, building_f, origin, config.context_steps, summary if summary_ok else None)
    if routing.mode == "reconstruct":
        st.info(f"{pretty(building_f)} is not forecast for this day: {routing.reason}. Its history is shown as "
                "bill-constrained reconstruction instead.", icon=":material/receipt_long:")
    else:
        out = day_ahead(files_signature, building_f, origin.isoformat(), model_choice,
                        provenance.file_fingerprint(selection.SUMMARY_PATH))
        frame = out["frame"]
        if out["fallback_reason"]:
            st.warning(f"{out['requested']} is unavailable ({out['fallback_reason']}); showing {out['used']} instead.",
                       icon=":material/sync_problem:")
        scores = pd.read_csv(selection.SUMMARY_PATH.parent / "scores.csv") if summary_ok and (selection.SUMMARY_PATH.parent / "scores.csv").exists() else None
        test_row = None
        if scores is not None:
            rows = scores[(scores["split"] == "test") & (scores["building"] == "ALL") & (scores["model"] == out["used"])]
            test_row = rows.iloc[0] if len(rows) else None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", out["used"], delta=out["version"], delta_color="off", delta_arrow="off", border=True, height="stretch")
        m2.metric("Forecast energy", f"{frame['predicted'].sum():,.0f} kWh", border=True, height="stretch")
        m3.metric("Benchmark WAPE (test)", f"{test_row.wape_pct:.1f}%" if test_row is not None else "not measured",
                  border=True, height="stretch", help="Weighted absolute % error on the held-out test month (synthetic data).")
        coverage = test_row.get("p10_p90_coverage") if test_row is not None else None
        m4.metric("P10-P90 coverage", f"{coverage:.0%}" if coverage is not None and pd.notna(coverage) else "n/a",
                  border=True, height="stretch", help="Share of actual readings inside the band; nominal 80%.")

        past = rec[(rec["building"] == building_f) & (rec["timestamp"] >= origin - pd.Timedelta(days=3))
                   & (rec["timestamp"] < origin)]
        fig = go.Figure()
        observed = past["kW"].where(past["is_observed"])
        fig.add_trace(go.Scatter(x=past["timestamp"], y=observed, name="Actual meter reading", mode="lines",
                                 line={"color": ACTUAL, "width": 1.6}))
        fig.add_trace(go.Scatter(x=past["timestamp"], y=past["kW"].where(past["is_reconstructed"]), name="Reconstructed",
                                 mode="lines", line={"color": RECONSTRUCTED, "width": 1.6, "dash": "dot"}))
        if {"q10", "q90"} <= set(frame.columns):
            fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["q90"] * 4, mode="lines", line={"width": 0},
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["q10"] * 4, mode="lines", line={"width": 0}, fill="tonexty",
                                     fillcolor="rgba(235,104,52,0.18)", name="Forecast P10-P90"))
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["predicted"] * 4, name=f"Forecast ({out['used']})", mode="lines",
                                 line={"color": FORECAST, "width": 2.2}))
        if frame["actual"].notna().any():
            fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["actual"] * 4, name="Actual (after the origin, for comparison)",
                                     mode="lines", line={"color": ACTUAL, "width": 1, "dash": "dash"}, opacity=0.6))
        fig.add_vline(x=origin, line_width=1, line_dash="dot", line_color="#888")
        top = axis_top(pd.concat([past["kW"], frame["predicted"] * 4, frame.get("q90", frame["predicted"]) * 4, frame["actual"] * 4]).dropna())
        chart_card(f"{pretty(building_f)} · next 24 hours from {origin:%d %b %Y}",
                   layout(fig, "Power (kW)", top, x_title="Date", peak_band=False),
                   f"Solid blue: real meter readings. Dotted grey: reconstructed (missing readings filled to match the bill). "
                   f"Orange: forecast. Routing: {routing.reason}. Scores come from synthetic data and are not real-world accuracy.")
