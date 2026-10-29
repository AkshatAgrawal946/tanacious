"""Streamlit dashboard for the Missing Meter MVP."""
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import ml_model, provenance, uncertainty
from src.data import load_data
from src.evaluate import score_series
from src.generate_data import generate
from src.reconstruct import reconstruct_all

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EVALUATION_META = PROCESSED_DIR / "evaluation_metrics.meta.json"
# Validated categorical palette (colour-blind safe, fixed order). Each building and each month
# keeps its slot, so changing the selection never repaints a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PEAK = "#e34948"
BASE_LOAD = "#9aa0a6"  # neutral grey: critical load we never touch
RANGE_FILL = "rgba(107,114,128,0.35)"
FLEX_LOAD = "#2a78d6"  # blue: weather-dependent load that is safe to shift
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
def evaluation_series(dataset_fingerprint: str, evaluation_signature: str) -> pd.DataFrame | None:
    """Per-interval predictions next to the truth from the last ``python -m src.pipeline`` run, only
    if that run scored this exact dataset (its metadata records the fingerprint)."""
    meta = provenance.read_json(EVALUATION_META)
    path = PROCESSED_DIR / "evaluation_15min.csv"
    if not path.exists() or meta is None or meta.get("dataset_fingerprint") != dataset_fingerprint:
        return None
    series = pd.read_csv(path, parse_dates=["timestamp"])
    return series.assign(month=series["timestamp"].dt.to_period("M").astype(str))


@st.cache_data
def tested_accuracy(dataset_fingerprint: str, evaluation_signature: str, months: tuple[str, ...]) -> pd.DataFrame | None:
    """Per-building accuracy over just the selected months."""
    series = evaluation_series(dataset_fingerprint, evaluation_signature)
    if series is None:
        return None
    scores = score_series(series[series["month"].isin(months)])
    return scores.set_index("building") if not scores.empty else None


@st.cache_data
def calibration(dataset_fingerprint: str, evaluation_signature: str) -> uncertainty.Calibration | None:
    """Likely-range ratios from the same test run as the accuracy badges."""
    series = evaluation_series(dataset_fingerprint, evaluation_signature)
    return None if series is None else uncertainty.calibrate(series)


def likely(low: float, high: float, unit: str = " kW", scale: float = 1.0, digits: int = 0) -> str:
    """'likely 90-108 kW'; 'metered' when nothing was imputed; '' when there is no tested range or it
    rounds to a single value (a mostly-metered building)."""
    if np.isnan(low) or np.isnan(high):
        return ""
    if np.isclose(low, high):
        return "metered"
    low_text, high_text = f"{low * scale:,.{digits}f}", f"{high * scale:,.{digits}f}"
    return "" if low_text == high_text else f"likely {low_text}-{high_text}{unit}"


def add_likely_band(fig: go.Figure, x, low, high, showlegend: bool = True, hover: bool = True) -> None:
    """Shaded P10-P90 range of the imputed load; nothing is drawn without a tested range."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if np.isnan(low).all() or np.allclose(low, high, equal_nan=True):
        return
    fig.add_trace(go.Scatter(x=x, y=high, mode="lines", line={"width": 0}, legendgroup="likely",
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=low, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=RANGE_FILL,
                             name="Likely range (P10-P90)", legendgroup="likely", showlegend=showlegend,
                             customdata=high, hoverinfo=None if hover else "skip",
                             hovertemplate="likely %{y:,.0f}-%{customdata:,.0f} kW<extra></extra>" if hover else None))


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
           height: int = 380, peak_band: bool = True) -> go.Figure:
    if peak_band:
        fig.add_vrect(x0=clock(16), x1=clock(20), fillcolor=PEAK, opacity=0.08, line_width=0,
                      annotation_text="4-8 PM peak", annotation_position="top left",
                      annotation_font={"size": 11, "color": PEAK})
    fig.update_layout(height=height, hovermode="x unified", margin={"t": 24, "b": 10, "l": 10, "r": 10},
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


def export_csv(frame: pd.DataFrame, temperature: pd.Series) -> bytes:
    """SHARE: the unified dataset (real readings + imputed intervals) as CSV,
    with a ``source`` column so finance and sustainability can tell measurement from estimate."""
    out = frame.assign(
        temperature_C=frame["timestamp"].map(temperature).round(1),
        source=np.where(frame["is_observed"], "metered", "imputed"),
    )
    columns = ["timestamp", "building", "building_type", "month", "kWh_consumed", "kW", "temperature_C",
               "source"]
    return out[columns].to_csv(index=False, float_format="%.4f").encode()


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
evaluation_signature = provenance.file_fingerprint(PROCESSED_DIR / "evaluation_15min.csv", EVALUATION_META)
cal = calibration(fingerprint, evaluation_signature)
# LIKELY RANGES: only the AI-imputed part of the load is uncertain. Real readings stay as measured,
# and bill totals are facts, so energy figures carry no range.
rec = rec.assign(kW_imputed=rec["kW"].where(~rec["is_observed"], 0.0))
rec["kW_low"], rec["kW_high"] = uncertainty.likely_range(cal and cal.interval, rec["kW_imputed"],
                                                         rec["kW"] - rec["kW_imputed"], rec["building"],
                                                         rec["timestamp"].dt.hour)
selected = rec[rec["month"].isin(months)]

campus_tab, building_tab = st.tabs([":material/apartment: Campus overview", ":material/domain: Building"])

with campus_tab:
    campus = selected.groupby("timestamp")["kW"].sum()
    energy = bills[bills["month"].isin(months)].groupby("building")["monthly_kWh"].sum()
    peak_window = selected["timestamp"].dt.hour.between(16, 19)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campus energy", f"{energy.sum() / 1000:,.1f} MWh", border=True, height="stretch",
              help="Sum of the monthly bills for the selected months.")
    evening_kwh = selected.loc[peak_window, ["kW", "kW_imputed"]].sum() / 4
    share_low, share_high = uncertainty.likely_range(cal and cal.peak_energy, evening_kwh["kW_imputed"],
                                                     evening_kwh["kW"] - evening_kwh["kW_imputed"], uncertainty.CAMPUS)
    total_kwh = selected["kWh_consumed"].sum()
    c2.metric("4-8 PM share", f"{evening_kwh['kW'] / total_kwh:.1%}", border=True, height="stretch",
              delta=likely(share_low / total_kwh, share_high / total_kwh, "%", 100, 1) or None,
              delta_color="off", delta_arrow="off",
              help="Likely range: how far a month's 4-8 PM energy has strayed from the model in testing (P10-P90).")
    peak_at = campus.idxmax()
    at_peak = selected.loc[selected["timestamp"] == peak_at, ["kW", "kW_imputed"]].sum()
    peak_low, peak_high = uncertainty.likely_range(cal and cal.interval, at_peak["kW_imputed"],
                                                   at_peak["kW"] - at_peak["kW_imputed"], uncertainty.CAMPUS, peak_at.hour)
    c3.metric("Campus peak", f"{campus.max():,.0f} kW",
              delta=" · ".join(filter(None, [peak_at.strftime("%d %b, %H:%M"), likely(peak_low, peak_high)])),
              delta_color="off", delta_arrow="off", border=True, height="stretch",
              help="Likely range: how far the campus total at this hour has strayed from the model in testing (P10-P90).")
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
    campus_day = average_day(selected.groupby("timestamp")[["kW", "kW_imputed"]].sum().reset_index(), ["kW", "kW_imputed"])
    add_likely_band(fig, clock(campus_day["hour"]), *uncertainty.likely_range(
        cal and cal.average_day, campus_day["kW_imputed"], campus_day["kW"] - campus_day["kW_imputed"],
        uncertainty.CAMPUS, campus_day["hour"].astype(int)))
    chart_card(f"Campus load on an average day · {period}",
               layout(fig, "Average power (kW)", axis_top(yearly)),
               "Each band is one building; the top edge is the whole campus, and the grey band around it is the "
               "likely range of the imputed load. "
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
    own_evening = own.loc[own["timestamp"].dt.hour.between(16, 19), ["kW", "kW_imputed"]].sum() / 4
    own_low, own_high = uncertainty.likely_range(cal and cal.peak_energy, own_evening["kW_imputed"],
                                                 own_evening["kW"] - own_evening["kW_imputed"], building)
    own_kwh = own["kWh_consumed"].sum()
    c2.metric("4-8 PM share", f"{own_evening['kW'] / own_kwh:.1%}", border=True, height="stretch",
              delta=likely(own_low / own_kwh, own_high / own_kwh, "%", 100, 1) or None,
              delta_color="off", delta_arrow="off",
              help="Likely range: how far a month's 4-8 PM energy has strayed from the model in testing (P10-P90).")
    c3.metric("Highest 15-min load", f"{peak_row.kW:,.0f} kW",
              delta=" · ".join(filter(None, [peak_row.timestamp.strftime("%d %b, %H:%M"), likely(peak_row.kW_low, peak_row.kW_high)])),
              delta_color="off", delta_arrow="off", border=True, height="stretch",
              help="Likely range: how far a single imputed 15-minute interval at this hour has strayed from the model "
                   "in testing (P10-P90). Real meter readings carry no range.")
    c4.metric("Real readings", f"{own['is_observed'].mean():.0%}", delta=meter_status[building],
              delta_color="off", delta_arrow="off", border=True, height="stretch")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        accuracy = tested_accuracy(fingerprint, evaluation_signature, tuple(months))
        if accuracy is not None and building in accuracy.index and own["is_reconstructed"].any():
            row = accuracy.loc[building]
            how = ("against the hidden true load" if row.evaluated_against == "hidden ground truth"
                   else "by rebuilding it with its own meter hidden")
            st.badge(f"{row.mape_pct:.1f}% avg. error per interval", icon=":material/target:", color="blue",
                     help=f"Tested over {period} {how}.")
            st.badge(f"{row.peak_window_error_pct:.1f}% error on 4-8 PM energy", icon=":material/target:", color="blue")

    # Base load (critical plug load, servers, lab equipment) is the floor the building never drops below:
    # the 5th percentile of 15-minute demand at any hour, capped at the average day's minimum so it is one
    # flat line under the whole curve (hostels dip lowest mid-day, when students are out, not overnight).
    # Everything above it is treated as flexible, weather-driven load (HVAC) that interventions may shift.
    day = average_day(own, ["kW", "kW_imputed"])
    base_kw = min(float(own["kW"].quantile(0.05)), float(day["kW"].min()))
    day["base"] = base_kw
    day["flexible"] = (day["kW"] - base_kw).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clock(day["hour"]), y=day["base"], name="Base Load (Critical)", stackgroup="load",
                             mode="lines", line={"width": 0.5, "color": BASE_LOAD}, fillcolor=BASE_LOAD,
                             hovertemplate="%{y:,.1f} kW<extra>Base (critical)</extra>"))
    fig.add_trace(go.Scatter(x=clock(day["hour"]), y=day["flexible"], name="Flexible Load (HVAC)", stackgroup="load",
                             mode="lines", line={"width": 0.5, "color": FLEX_LOAD}, fillcolor=FLEX_LOAD,
                             hovertemplate="%{y:,.1f} kW<extra>Flexible (HVAC)</extra>"))
    add_likely_band(fig, clock(day["hour"]), *uncertainty.likely_range(
        cal and cal.average_day, day["kW_imputed"], day["kW"] - day["kW_imputed"], building, day["hour"].astype(int)))
    own_year = rec[rec["building"] == building]
    chart_card(f"{pretty(building)} · base vs. flexible load on an average day · {period}",
               layout(fig, "Average power (kW)", axis_top(average_day(own_year, ["kW"], ["month"])["kW"])),
               "Base load is the 5th percentile of 15-minute demand, held flat across the day; "
               "everything above it is weather-dependent load. The grey band is the likely range of the imputed load.")
    peak = day.loc[day["kW"].idxmax()]
    peak_time = f"{int(peak.hour):02d}:{int(peak.hour % 1 * 60):02d}"
    st.info(f"**At the average-day peak ({peak_time}), {peak.base:,.0f} kW is critical base load** that stays untouched; "
            f"**{peak.flexible:,.0f} kW ({peak.flexible / peak.kW:.0%} of the peak) is flexible cooling/HVAC load** "
            "that is safe for load-shifting interventions.", icon=":material/shield:")

    fig = go.Figure()
    for i, m in enumerate(months):
        rows = own[own["month"] == m]
        # Drawn per month so the band never bridges the gap between non-adjacent months.
        add_likely_band(fig, rows["timestamp"], rows["kW_low"], rows["kW_high"], showlegend=i == 0, hover=False)
    for m in months:
        rows = own[own["month"] == m]
        note = np.where(rows["is_observed"], " (metered)", [f" (likely {lo:,.0f}-{hi:,.0f})" if hi == hi else ""
                                                          for lo, hi in zip(rows["kW_low"], rows["kW_high"])])
        fig.add_trace(go.Scattergl(x=rows["timestamp"], y=rows["kW"], name=month_label(m), mode="lines",
                                   line={"width": 1, "color": month_colors[m]}, text=note,
                                   hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kW%{text}<extra></extra>"))
    year_own = rec[rec["building"] == building]
    chart_card(f"{pretty(building)} · 15-minute load",
               layout(fig, "Power (kW)", axis_top(year_own["kW_high"].fillna(year_own["kW"])),
                      x_title="Date", peak_band=False),
               f"{int(own['is_observed'].sum()):,} of {len(own):,} intervals are real meter readings; the rest are "
               "reconstructed and scaled so each month matches its bill. The grey band is where the true load "
               "likely sits (P10-P90 of the model's tested error); metered stretches have no band.")

    chart_card(f"{pretty(building)} · monthly energy", monthly_bars(bills, [building], months, building_colors),
               "Selected months are shown in full colour.")

# SHARE: rendered last so it sits at the bottom of the sidebar. The CSV is built only when clicked
# (``data`` is a callable).
with st.sidebar:
    st.divider()
    scope = st.segmented_control("Export scope", ["Selected months", "Full year"], default="Selected months",
                                 required=True, key="export_scope")
    export_rows = selected if scope == "Selected months" else rec
    temperature = ml_model.campus_temperature(interval)
    st.download_button("Export Data Report", icon="⬇️", type="primary", width="stretch",
                       data=lambda: export_csv(export_rows, temperature),
                       file_name=f"missing_meter_{'_'.join(months) if scope == 'Selected months' else 'full_year'}.csv",
                       mime="text/csv", on_click="ignore",
                       help="Metered readings + AI-imputed intervals for every building, "
                            "with a 'source' column marking which is which.")
    st.caption(f"{len(export_rows):,} rows · {export_rows['is_observed'].mean():.0%} metered")
