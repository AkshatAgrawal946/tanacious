"""Streamlit dashboard for the Missing Meter MVP."""
import csv
import hashlib
import io
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

from src import backtest, ml_model, provenance, simulate, uncertainty
from src.data import load_data
from src.evaluate import score_series
from src.generate_data import generate
from src.reconstruct import reconstruct_all

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EVALUATION_META = PROCESSED_DIR / "evaluation_metrics.meta.json"
REAL_METER_FILE = Path("data/real/fox_campus_2017.csv")  # see data/real/README.md (BDG2, CC BY-SA 4.0)
# Validated categorical palette (colour-blind safe, fixed order). Each building and each month
# keeps its slot, so changing the selection never repaints a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PEAK = "#e34948"
BASE_LOAD = "#9aa0a6"  # neutral grey: critical load we never touch
RANGE_FILL = "rgba(107,114,128,0.35)"
FLEX_LOAD = "#2a78d6"  # blue: weather-dependent load that is safe to shift
# Gain/loss in the Simulate metrics and table: red for a loss, grey for no change, green for a gain.
# greenColor/redColor in .streamlit/config.toml match, so metric deltas use the same shades.
GAIN, LOSS, NO_CHANGE = "#1baf7a", PEAK, BASE_LOAD
MAX_MONTHS = len(SERIES)
CURRENCY = "₹"  # tariff currency in the Simulate tab; edit with the rates in simulate.Tariff

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


# TEST ON YOUR DATA: upload real meter readings, hide some, let the model rebuild them, compare.
# Every step is cached on the file's bytes and the settings, so changing one control re-runs only
# what depends on it (the forest is retrained only when the hidden readings or types change).
@st.cache_data
def sample_meter_csv(files_signature: str) -> bytes:
    """The dashboard's two smart-metered buildings as a typical meter export (day-first dates, one
    column per meter, a temperature column). Synthetic: it only shows how the test works."""
    interval, _, _ = campus_data(files_signature)
    wide = interval[interval["building"].isin(["B3_Academic", "B5_Lab"])].pivot(
        index="timestamp", columns="building", values="kWh_consumed")
    temperature = ml_model.campus_temperature(interval).reindex(wide.index)
    return pd.DataFrame({"Timestamp": wide.index.strftime("%d/%m/%Y %H:%M"),
                         "Academic Block kWh": wide["B3_Academic"].round(3).to_numpy(),
                         "Research Lab kWh": wide["B5_Lab"].round(3).to_numpy(),
                         "Outdoor Temp (C)": temperature.round(1).to_numpy()}).to_csv(index=False).encode()


@st.cache_data
def read_meter_csv(content: bytes) -> pd.DataFrame:
    """Any delimiter (``,`` ``;`` tab ``|``, sniffed from the start of the file, so pandas' fast parser
    reads even a 175 MB export) and UTF-8 or Latin-1 text."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            head = content[:65536].decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            sep = csv.Sniffer().sniff(head.split("\n", 1)[0], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
        return pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
    raise ValueError("The file is not readable text.")


@st.cache_data
def meter_file_guess(content: bytes) -> dict:
    return backtest.guess_columns(read_meter_csv(content))


@st.cache_data
def prepared_meters(content: bytes, timestamp: str, meters: tuple[str, ...], unit: str, building: str | None,
                    temperature: str | None, dayfirst: bool, keep: tuple[str, ...] | None) -> backtest.MeterData:
    return backtest.prepare(read_meter_csv(content), timestamp, list(meters), unit, building, temperature, dayfirst,
                            None if keep is None else list(keep))


@st.cache_data(max_entries=20)
def rebuilt_meter(content: bytes, settings: tuple, types: tuple[tuple[str, str], ...], building: str,
                  hidden: bytes, files_signature: str, dataset_fingerprint: str) -> tuple[pd.DataFrame, str]:
    data = prepared_meters(content, *settings)
    mask = np.frombuffer(hidden, dtype=bool)
    interval, bills, _ = campus_data(files_signature)
    # Only needed when every reading in the file is hidden: the campus-trained forest then predicts
    # the shape from the building type alone.
    campus_shape = ml_model.shape_fn(forest(files_signature, dataset_fingerprint), interval, bills)
    return backtest.rebuild(data, dict(types), building, mask, fallback=campus_shape)


# Building types as people say them; "Leave out" drops a meter (parking, offices, shops...).
TYPE_LABELS = {"hostel": "Hostel / dorm", "academic": "Academic / classroom", "lab": "Lab", "library": "Library"}
LEAVE_OUT = "Leave out"
MAX_TEST_METERS = 10  # the forest trains on every kept meter; more than this makes a rebuild slow


def readings_every(resolution: pd.Timedelta) -> str:
    minutes = int(resolution / pd.Timedelta("1min"))
    return "Hourly readings" if minutes == 60 else f"{minutes}-minute readings"


def per_period(series: pd.DataFrame, resolution: pd.Timedelta) -> pd.DataFrame:
    """15-minute results at the file's own resolution, as average power (kW)."""
    periods = series.assign(scored=series["hidden"] & ~series["filled"]).groupby(series.index.floor(resolution)).agg(
        actual=("actual", "sum"), predicted=("predicted", "sum"), flat=("flat", "sum"), hidden=("scored", "all"))
    hours = resolution / pd.Timedelta("1h")
    return periods.assign(**{c: periods[c] / hours for c in ("actual", "predicted", "flat")})


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

campus_tab, building_tab, simulate_tab, test_tab = st.tabs([
    ":material/apartment: Campus overview", ":material/domain: Building", ":material/tune: Simulate",
    ":material/science: Test on your data"])

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

    # Base load (critical plug load, servers, lab equipment) is the floor the building never drops below;
    # everything above it is treated as flexible, weather-driven load (HVAC) that interventions may shift.
    # The Simulate tab uses the same split.
    day = average_day(own, ["kW", "kW_imputed"])
    base_kw = simulate.base_load(own["kW"], hour_of_day(own["timestamp"]))
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
        note = np.where(rows["is_observed"], " (metered)", [f" (likely {lo:,.0f}-{hi:,.0f})" if not np.isnan(hi) else ""
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

with simulate_tab:
    st.caption("What if part of each building's flexible (HVAC) load were pre-cooled earlier in the day, or trimmed "
               "with a setpoint change? Base load is never touched, and every day keeps its own energy apart from "
               "the pre-cooling loss and the trim.")
    with st.container(border=True):
        st.markdown("##### Scenario")
        taking_part = st.multiselect("Buildings taking part", buildings, default=buildings, format_func=pretty)
        s1, s2 = st.columns(2)
        preset = simulate.Scenario()  # the defaults found by searching the year (see Scenario)
        shift = s1.slider("Move flexible 4-8 PM load", 0, 100, round(preset.shift_share * 100), 5, format="%d%%",
                          help="Share of the load above each building's base load, inside 4-8 PM, that is "
                               "pre-cooled earlier instead.")
        window = s2.slider("Pre-cool between", 0, simulate.PEAK_HOURS[0], preset.target_hours, format="%d:00",
                           help="Moved load fills this window's quietest 15 minutes first, so it does not build a new peak.")
        loss = s1.slider("Pre-cooling loss", 0, 50, round(preset.shift_loss * 100), 5, format="%d%%",
                         help="Extra energy per kWh moved: a pre-cooled building leaks some of it back.")
        trim = s2.slider("Trim flexible load", 0, 30, round(preset.trim * 100), 1, format="%d%%",
                         help="Removed from flexible load at every hour, e.g. by raising the cooling setpoint "
                              "(often quoted as a few percent of cooling energy per 1 °C).")
        with st.expander(f"Tariff ({CURRENCY}, placeholder rates: use the campus's own)"):
            t1, t2, t3 = st.columns(3)
            default = simulate.Tariff()
            tariff = simulate.Tariff(
                off_peak=t1.number_input("Off-peak, per kWh", 0.0, value=default.off_peak, step=0.5),
                peak=t2.number_input("4-8 PM, per kWh", 0.0, value=default.peak, step=0.5),
                demand=t3.number_input("Demand, per kW-month", 0.0, value=default.demand, step=25.0,
                                       help="Charged on each building's highest 15-minute load in each month."))

    if not taking_part:
        st.info("Pick at least one building to simulate.", icon=":material/domain:")
    else:
        scenario = simulate.Scenario(shift_share=shift / 100, target_hours=window, shift_loss=loss / 100, trim=trim / 100)
        sim = simulate.simulate(selected[["timestamp", "building", "kW"]], scenario, taking_part)
        campus_sim = sim.groupby("timestamp")[["kW", "kW_after"]].sum()
        in_peak = simulate.in_window(sim["timestamp"])
        # Rounded (and -0.0 turned into 0.0) so a null scenario reads as "+0", not "-0.0 MWh".
        moved_kwh = round((sim.loc[in_peak, "kW"] - sim.loc[in_peak, "kW_after"]).sum() / 4) + 0.0
        energy_change = round((sim["kW_after"] - sim["kW"]).sum() / 4) + 0.0
        bill_before = simulate.cost(sim["timestamp"], sim["building"], sim["kW"], tariff)
        bill_after = simulate.cost(sim["timestamp"], sim["building"], sim["kW_after"], tariff)
        before_total, after_total = bill_before.to_numpy().sum(), bill_after.to_numpy().sum()

        def change(change: float, before: float, text: str) -> dict:
            """Metric delta arguments for a change where up is a loss (red) and down a gain (green). One that
            rounds to 0 in its unit (kW, kWh, currency) is passed as the number 0, the only delta Streamlit
            draws grey."""
            if round(change) == 0:
                return {"delta": 0, "delta_description": "no change", "delta_color": "inverse"}
            return {"delta": f"{text} ({change / before:+.1%})", "delta_color": "inverse"}

        def money(amount: float) -> str:
            """'-₹8,827': the sign goes first, since Streamlit reads a delta's direction from its first character."""
            return f"{'-' if amount < 0 else '+'}{CURRENCY}{abs(amount):,.0f}"

        c1, c2, c3, c4 = st.columns(4)
        peak_before, peak_after = campus_sim["kW"].max(), campus_sim["kW_after"].max()
        c1.metric("Campus peak", f"{peak_after:,.0f} kW", border=True, height="stretch",
                  **change(peak_after - peak_before, peak_before, f"{peak_after - peak_before:+,.0f} kW"),
                  help="Highest 15-minute campus load over the selected months, after the scenario.")
        c2.metric("Moved out of 4-8 PM", f"{moved_kwh / 1000:,.1f} MWh",
                  delta=f"{moved_kwh / (sim.loc[in_peak, 'kW'].sum() / 4):.0%} of 4-8 PM energy",
                  delta_color="off", delta_arrow="off", border=True, height="stretch")
        c3.metric("Energy use", f"{(sim['kW_after'].sum() / 4) / 1000:,.1f} MWh", border=True, height="stretch",
                  **change(energy_change, sim["kW"].sum() / 4, f"{energy_change:+,.0f} kWh"),
                  help="Trim savings minus pre-cooling losses.")
        c4.metric("Electricity cost", f"{CURRENCY}{after_total:,.0f}", border=True, height="stretch",
                  **change(after_total - before_total, before_total, money(after_total - before_total)),
                  help="Time-of-use energy charges plus demand charges, at the tariff above.")

        def before_after(fig: go.Figure, x, before, after) -> None:
            fig.add_trace(go.Scatter(x=x, y=before, name="Today", mode="lines",
                                     line={"width": 1.5, "color": BASE_LOAD, "dash": "dash"},
                                     hovertemplate="%{y:,.0f} kW<extra>Today</extra>"))
            fig.add_trace(go.Scatter(x=x, y=after, name="With scenario", mode="lines",
                                     line={"width": 2, "color": FLEX_LOAD},
                                     hovertemplate="%{y:,.0f} kW<extra>With scenario</extra>"))

        def precool_band(fig: go.Figure, x0, x1) -> None:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=FLEX_LOAD, opacity=0.08, line_width=0,
                          annotation_text="pre-cool", annotation_position="top left",
                          annotation_font={"size": 11, "color": FLEX_LOAD})

        sim_day = average_day(campus_sim.reset_index(), ["kW", "kW_after"])
        fig = go.Figure()
        before_after(fig, clock(sim_day["hour"]), sim_day["kW"], sim_day["kW_after"])
        precool_band(fig, clock(window[0]), clock(window[1]))
        chart_card(f"Campus load on an average day · {period}",
                   layout(fig, "Average power (kW)", axis_top(sim_day[["kW", "kW_after"]].max())),
                   "Dashed: reconstructed load today. Solid: the same days with the scenario applied.")

        peak_day = campus_sim["kW"].idxmax().normalize()
        worst = campus_sim[(campus_sim.index >= peak_day) & (campus_sim.index < peak_day + pd.Timedelta(days=1))]
        fig = go.Figure()
        before_after(fig, clock(hour_of_day(worst.index.to_series())), worst["kW"], worst["kW_after"])
        precool_band(fig, clock(window[0]), clock(window[1]))
        chart_card(f"Campus peak day · {peak_day:%a %d %b}",
                   layout(fig, "Power (kW)", axis_top(worst[["kW", "kW_after"]].max())),
                   "The day with the highest 15-minute campus load. The demand charge is set by days like this, "
                   "so this is where shaving the peak pays.")

        by_building = sim.groupby("building")
        per_building = by_building.agg(base=("base_kW", "first"), before=("kW", "max"))
        per_building["peak_change"] = by_building["kW_after"].max() - per_building["before"]
        per_building["moved"] = (sim[in_peak].groupby("building")[["kW", "kW_after"]].sum()
                                 .pipe(lambda f: (f["kW"] - f["kW_after"]) / 4))
        per_building["energy_change"] = (by_building["kW_after"].sum() - by_building["kW"].sum()) / 4
        per_building["cost_change"] = bill_after.sum(axis=1) - bill_before.sum(axis=1)
        per_building.index = per_building.index.map(pretty)
        changes = ["peak_change", "energy_change", "cost_change"]
        # Rounded first so a change that displays as 0 is grey rather than a red or green "0".
        per_building[changes] = per_building[changes].round().add(0.0)

        def gain_or_loss(value: float) -> str:
            return f"color: {LOSS if value > 0 else GAIN if value < 0 else NO_CHANGE}"

        with st.container(border=True):
            st.markdown("##### Buildings in the scenario")
            st.dataframe(per_building.rename_axis("Building").reset_index().style.map(gain_or_loss, subset=changes),
                         width="stretch", hide_index=True,
                         column_config={
                             "base": st.column_config.NumberColumn("Base load", format="%.0f kW"),
                             "before": st.column_config.NumberColumn("Peak today", format="%.0f kW"),
                             "peak_change": st.column_config.NumberColumn("Peak change", format="%+.0f kW"),
                             "moved": st.column_config.NumberColumn("Moved out of 4-8 PM", format="%.0f kWh"),
                             "energy_change": st.column_config.NumberColumn("Energy change", format="%+.0f kWh"),
                             "cost_change": st.column_config.NumberColumn(f"Cost change ({CURRENCY})", format="%+.0f"),
                         })
            st.caption("Red is a loss (more load or cost), green a gain, grey no change. Buildings not taking part "
                       "show no base load and no change. For bill-only buildings the "
                       "flexible load is itself reconstructed, so treat their figures as estimates (see the likely "
                       "ranges in the Building tab).")

with test_tab:
    def test_on_your_data() -> None:
        st.caption("Upload real meter readings, hide some of them, and let the model rebuild what you hid from the "
                   "rest of the file and each month's total, just as it rebuilds a bill-only building. Then compare "
                   "the rebuilt values with the real readings you hid.")

        with st.container(border=True):
            st.markdown("##### 1 · Meter data")
            upload = st.file_uploader("CSV of meter readings", type=["csv", "txt"],
                                      help="A timestamp column plus one column per meter, or a building-name column and "
                                           "a value column. Readings every 1-15 minutes, 30 minutes or hourly.")
            if upload is None:
                st.caption("No file to hand? Use one of these:")
                with st.container(horizontal=True, gap="small"):
                    if REAL_METER_FILE.exists() and st.button("Real campus data (US university, 2017)",
                                                              icon=":material/apartment:"):
                        st.session_state["bt_source"] = "real"
                    if st.button("Synthetic sample", icon=":material/science:"):
                        st.session_state["bt_source"] = "synthetic"
            source = "upload" if upload is not None else st.session_state.get("bt_source")
            content = {"upload": lambda: upload.getvalue(), "real": REAL_METER_FILE.read_bytes,
                       "synthetic": lambda: sample_meter_csv(files_signature)}.get(source, lambda: None)()
            if content is None:
                st.info("Upload a CSV to begin.", icon=":material/upload_file:")
                return
            if source == "real":
                st.caption("Seven buildings of a US university campus, hourly, 2017, from the Building Data Genome "
                           "Project 2 (CC BY-SA 4.0); see data/real/README.md.")
            elif source == "synthetic":
                st.caption("Synthetic (the dashboard's Academic block and Lab): it shows how the test works, not how "
                           "the model does on real data.")
            try:
                raw = read_meter_csv(content)
            except (ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not read the file: {error}")
                return
            guess = meter_file_guess(content)
            if not guess["numeric"]:
                st.error("No column of numbers found: is this a meter export?")
                return
            file_key = hashlib.sha1(content).hexdigest()[:10]  # new widgets (and defaults) for a new file
            columns = list(raw.columns)

            # The file is read with the guessed settings, summarised in one line.
            timestamp = guess["timestamp"] or columns[0]
            unit = backtest.UNITS[0]
            numeric = guess["numeric"]
            temperature = guess["temperature"]
            building_column = value_column = None
            if guess["building"] and len(guess["meters"]) == 1:
                labels = [c for c in columns if c not in numeric and c != timestamp] or columns
                building_column = guess["building"] if guess["building"] in labels else labels[0]
                value_column = numeric[0]
            dayfirst = guess["dayfirst"]
            if building_column:
                candidates = sorted(raw[building_column].dropna().astype(str).str.strip().unique())
            else:
                candidates = [c for c in numeric if c not in (timestamp, temperature)]
            st.markdown(
                f"Found **{len(candidates):,} meter{'s' if len(candidates) != 1 else ''}**"
                + (f" in **{building_column}**" if building_column else "")
                + f" · times from **{timestamp}** · values read as **{unit[0].lower() + unit[1:]}** · "
                + (f"temperature from **{temperature}**" if temperature else "no temperature column"))

        with st.container(border=True):
            st.markdown("##### 2 · Buildings")
            st.caption("Tell the model what each meter is, so it knows the daily routine to expect. Anything that is "
                       f"not a hostel, classroom, lab or library (parking, offices, shops...) is best left out. Up to "
                       f"{MAX_TEST_METERS} meters: the model learns from every meter you keep, and you test one of them.")
            guessed = {name: backtest.guess_type(name) for name in candidates}
            recognised = [n for n in candidates if guessed[n]]
            table = pd.DataFrame({"Meter": recognised + [n for n in candidates if not guessed[n]]})
            table["Building type"] = [TYPE_LABELS[guessed[n]] if i < MAX_TEST_METERS and guessed[n] else LEAVE_OUT
                                      for i, n in enumerate(table["Meter"])]
            edited = st.data_editor(
                table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 330),
                key=f"bt_types_{file_key}_{building_column}_{temperature}",
                column_config={"Meter": st.column_config.TextColumn(disabled=True),
                               "Building type": st.column_config.SelectboxColumn(
                                   options=[*TYPE_LABELS.values(), LEAVE_OUT], required=True)})
            kept = edited[edited["Building type"] != LEAVE_OUT]
            if kept.empty:
                st.info("Give at least one meter a building type.", icon=":material/domain:")
                return
            if len(kept) > MAX_TEST_METERS:
                st.warning(f"{len(kept)} meters have a type; set some to {LEAVE_OUT} to keep at most {MAX_TEST_METERS}.")
                return
            label_to_type = {label: kind for kind, label in TYPE_LABELS.items()}
            types = {meter: label_to_type[label] for meter, label in zip(kept["Meter"], kept["Building type"])}
            names = tuple(types)
            settings = (timestamp, (value_column,) if building_column else names, unit, building_column, temperature,
                        dayfirst, names if building_column else None)
            try:
                data = prepared_meters(content, *settings)
            except ValueError as error:
                st.error(str(error))
                return
            readings = data.readings
            usable = [n for n in names if n in set(readings["building"])]
            if len(usable) < len(names):
                st.warning("Not enough complete months to use: " + ", ".join(n for n in names if n not in usable))
            building = st.selectbox("Building to test", usable, key=f"bt_building_{file_key}",
                                    format_func=lambda n: f"{n} ({TYPE_LABELS[types[n]].lower()})")
            own = readings[readings["building"] == building]
            first, last = own["timestamp"].min(), own["timestamp"].max()
            months = own["timestamp"].dt.to_period("M").nunique()
            st.caption(f"{readings_every(data.resolution)} · {first:%b %Y} to {last:%b %Y} · {months} complete "
                       f"month{'s' if months != 1 else ''} used")
            if data.notes:
                with st.expander(f"Data notes ({len(data.notes)})"):
                    for note in data.notes:
                        st.markdown(f"- {note}")

        with st.container(border=True):
            st.markdown("##### 3 · Readings to hide")
            mode = st.segmented_control("Hide", ["Random days", "A date range", "Same hours every day", "Draw on the chart"],
                                        default="Random days", required=True, key="bt_mode")
            ts = own["timestamp"]
            hourly = own.groupby(own["timestamp"].dt.floor("h"))["kWh_consumed"].sum()
            if mode == "Random days":
                share = st.slider("Share of days hidden", 5, 90, 30, 5, format="%d%%", key="bt_share")
                hidden = backtest.hide_random_days(ts, share / 100)
            elif mode == "A date range":
                default_end = min(first.date() + pd.Timedelta(days=13), last.date())
                picked = st.date_input("Hide from / to (inclusive)", (first.date() + pd.Timedelta(days=7), default_end),
                                       min_value=first.date(), max_value=last.date(), key=f"bt_range_{file_key}")
                if len(picked) != 2:
                    st.info("Pick an end date.")
                    return
                hidden = backtest.hide_between(ts, picked[0], pd.Timestamp(picked[1]) + pd.Timedelta(days=1))
            elif mode == "Same hours every day":
                hours = st.slider("Hours hidden every day", 0, 24, (16, 20), format="%d:00", key="bt_hours")
                hidden = backtest.hide_hours(ts, *hours)
                st.caption("Only these hours are hidden, so each month's total fixes their combined energy exactly; "
                           "the test is how the model spreads it over the days and intervals.")
            else:
                fig = go.Figure(go.Scattergl(x=hourly.index, y=hourly, mode="lines", name="Real readings",
                                             line={"width": 1, "color": FLEX_LOAD},
                                             hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kWh<extra></extra>"))
                fig = layout(fig, "Energy per hour (kWh)", axis_top(hourly), x_title="Date", peak_band=False)
                fig.update_layout(dragmode="select", selectdirection="h")
                event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="box",
                                        key=f"bt_draw_{file_key}_{building}", config={"displaylogo": False})
                boxes = event.selection.get("box", []) if event else []
                hidden = np.zeros(len(ts), dtype=bool)
                for box in boxes:
                    start, end = sorted(pd.to_datetime(box["x"]))
                    hidden |= backtest.hide_between(ts, start, end)
                st.caption("Drag across the chart to hide those readings; hold Shift to add more stretches. "
                           "Double-click to clear.")
            hidden_kwh = own.loc[hidden, "kWh_consumed"].sum()
            st.markdown(f"**{int(hidden.sum()):,} of {len(own):,} slots hidden** · "
                        f"{hidden_kwh / own['kWh_consumed'].sum():.0%} of this meter's energy")
            if mode != "Draw on the chart":
                shown = own.assign(hidden=hidden).groupby(own["timestamp"].dt.floor("h")).agg(
                    kWh=("kWh_consumed", "sum"), hidden=("hidden", "any"))
                fig = go.Figure()
                fig.add_trace(go.Scattergl(x=shown.index, y=shown["kWh"].where(~shown["hidden"]), mode="lines",
                                           name="Kept", line={"width": 1, "color": FLEX_LOAD},
                                           hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kWh<extra>Kept</extra>"))
                fig.add_trace(go.Scattergl(x=shown.index, y=shown["kWh"].where(shown["hidden"]), mode="lines",
                                           name="Hidden from the model", line={"width": 1, "color": BASE_LOAD},
                                           hovertemplate="%{x|%d %b %H:%M}: %{y:,.1f} kWh<extra>Hidden</extra>"))
                st.plotly_chart(layout(fig, "Energy per hour (kWh)", axis_top(shown["kWh"]), x_title="Date",
                                       peak_band=False), width="stretch", config={"displaylogo": False})

        request = (file_key, settings, tuple(types.items()), building, hashlib.sha1(hidden.tobytes()).hexdigest())
        ready = st.session_state.get("bt_request") == request
        if st.button("Rebuild the hidden readings", type="primary", icon=":material/auto_fix_high:",
                     disabled=not hidden.any()):
            st.session_state["bt_request"], ready = request, True
        if not hidden.any():
            st.info("Hide some readings first.", icon=":material/visibility_off:")
            return
        if not ready:
            return
        with st.spinner("Training on the readings you kept and rebuilding the rest..."):
            try:
                series, trained_on = rebuilt_meter(content, settings, tuple(types.items()), building, hidden.tobytes(),
                                                   files_signature, fingerprint)
                scores = backtest.score(series, data.resolution)
            except ValueError as error:
                st.error(str(error))
                return

        model, flat = scores.loc["model"], scores.loc["flat_bill_split"]
        st.markdown("##### 4 · How close did it get?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hidden readings", f"{int(model.scored_periods):,}", border=True, height="stretch",
                  help="Hidden periods scored at the file's own resolution (filled gaps are never scored).")
        gap = model.wape_pct - flat.wape_pct
        c2.metric("Model error", f"{model.wape_pct:.1f}%", border=True, height="stretch", delta_color="inverse",
                  delta=f"{gap:+.1f} pts vs. flat split" if round(gap, 1) else 0,
                  help="WAPE: total absolute error over the total real energy of the hidden readings.")
        c3.metric("Flat bill split error", f"{flat.wape_pct:.1f}%", border=True, height="stretch",
                  help="The month's remaining energy spread evenly over the hidden readings: the no-model baseline.")
        c4.metric("4-8 PM energy error", "n/a" if np.isnan(model.peak_window_error_pct) else f"{model.peak_window_error_pct:.1f}%",
                  border=True, height="stretch", help="Hidden 4-8 PM energy, model vs. real.")
        st.caption(f"Trained on {trained_on}. Each month still adds up to its total, and every kept reading comes "
                   "back unchanged.")

        periods = per_period(series, data.resolution)
        hidden_periods = periods["hidden"]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=periods.index, y=periods["actual"], mode="lines", name="Real readings",
                                   line={"width": 1.5, "color": BASE_LOAD},
                                   hovertemplate="%{y:,.1f} kW<extra>Real</extra>"))
        fig.add_trace(go.Scattergl(x=periods.index, y=periods["predicted"].where(hidden_periods), mode="lines",
                                   name="Model (hidden readings)", line={"width": 1.5, "color": FLEX_LOAD},
                                   hovertemplate="%{y:,.1f} kW<extra>Model</extra>"))
        fig.add_trace(go.Scattergl(x=periods.index, y=periods["flat"].where(hidden_periods), mode="lines",
                                   name="Flat bill split", visible="legendonly", line={"width": 1.5, "color": SERIES[1]},
                                   hovertemplate="%{y:,.1f} kW<extra>Flat split</extra>"))
        # Zoom to the 8 days where the model is closest on the hidden readings, among windows holding at
        # least half as much hidden energy as the fullest one (so a few lucky hours cannot win).
        scored = periods[hidden_periods]
        days = pd.date_range(periods.index[0].floor("D"), periods.index[-1].floor("D"), freq="D")
        error = (scored["predicted"] - scored["actual"]).abs().groupby(scored.index.floor("D")).sum().reindex(days, fill_value=0)
        energy = scored["actual"].groupby(scored.index.floor("D")).sum().reindex(days, fill_value=0)
        window_error, window_energy = error.rolling(8, min_periods=1).sum(), energy.rolling(8, min_periods=1).sum()
        best_end = (window_error / window_energy).where(window_energy >= window_energy.max() / 2).fillna(np.inf).idxmin()
        start = max(best_end - pd.Timedelta(days=7), periods.index[0])
        fig = layout(fig, "Power (kW)", axis_top(periods[["actual", "predicted"]].max()), x_title="Date", peak_band=False)
        fig.update_xaxes(range=[start, min(best_end + pd.Timedelta(days=1), periods.index[-1])])
        chart_card(f"{building} · real vs. rebuilt", fig,
                   "Zoomed to the 8 days where the model came closest on the hidden readings; drag to pan, "
                   "double-click to see everything. Blue appears only "
                   "where readings were hidden. Click 'Flat bill split' in the legend to compare with the no-model baseline.")

        day = average_day(periods[hidden_periods].rename_axis("timestamp").reset_index(), ["actual", "predicted", "flat"])
        fig = go.Figure()
        for column, name, color, dash in (("actual", "Real readings", BASE_LOAD, "dash"), ("predicted", "Model", FLEX_LOAD, None),
                                          ("flat", "Flat bill split", SERIES[1], "dot")):
            fig.add_trace(go.Scatter(x=clock(day["hour"]), y=day[column], name=name, mode="lines",
                                     line={"width": 2, "color": color, "dash": dash},
                                     hovertemplate="%{y:,.1f} kW<extra>" + name + "</extra>"))
        chart_card("Hidden readings on an average day", layout(fig, "Average power (kW)", axis_top(day[["actual", "predicted"]].max())),
                   "The average of every hidden reading by time of day: does the model put the energy at the right hours?")

        out = series.rename_axis("timestamp").reset_index()
        st.download_button("Download the comparison (CSV)", icon=":material/download:",
                           data=out.to_csv(index=False, float_format="%.4f").encode(),
                           file_name=f"backtest_{building}.csv".replace(" ", "_"), mime="text/csv", on_click="ignore")

    test_on_your_data()

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
