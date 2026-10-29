"""Test the model on your own meter data: hide some real readings, rebuild them, compare.

1. ``guess_columns`` / ``prepare``: read any meter CSV (one column per meter, or a building column),
   in kWh per interval, kW, or a cumulative kWh register, at any interval that divides or is a
   multiple of 15 minutes, onto the model's 15-minute grid.
2. Each month's total stands in for the bill. Only months with at least ``MIN_COVERAGE`` of their
   readings are kept, so the total is (almost) the real one; the few missing intervals are filled
   by interpolation and never scored or trained on.
3. ``hide_*``: choose readings to hide in one building.
4. ``rebuild``: the same Random Forest recipe as the dashboard (``ml_model.train``), trained only on
   the readings left visible in the file, reconstructs the hidden ones with the monthly totals
   (``reconstruct_building_month``). If nothing in the file is left visible, a fallback shape (the
   campus-trained forest) is used. The flat bill split is rebuilt alongside as a baseline.
5. ``score``: hidden real readings vs. model vs. flat split, at the file's own resolution: hourly
   data is spread over four 15-minute slots for the model, and scored hourly again, so the model is
   never marked against values we made up.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import ml_model
from src.evaluate import flat_shape
from src.metrics import PEAK_HOURS, score_prediction
from src.reconstruct import ShapeFn, reconstruct_building_month

FREQ = pd.Timedelta("15min")
MIN_COVERAGE = 0.95  # share of a month's readings needed for its total to stand in for the bill
NEUTRAL_TEMPERATURE = 22.0  # used when the file has no temperature: no cooling load is modelled
KWH, KW, REGISTER = "Energy used in each interval (kWh)", "Average power (kW)", "Running meter total (kWh)"
UNITS = (KWH, KW, REGISTER)
# Words in a meter's name that give away its building type (matched at the start of each word, so
# "Classroom" -> academic, "Fox_lodging_Cora" -> hostel). Anything else gets no type: parking, offices,
# shops and the like follow schedules the model was never given, so they are left out by default.
TYPE_WORDS = (("hostel", "hostel"), ("dorm", "hostel"), ("lodging", "hostel"), ("residen", "hostel"),
              ("librar", "library"), ("lab", "lab"), ("research", "lab"), ("science", "lab"),
              ("class", "academic"), ("academ", "academic"), ("educat", "academic"), ("lecture", "academic"),
              ("school", "academic"), ("teach", "academic"))


@dataclass
class MeterData:
    readings: pd.DataFrame  # timestamp, building, kWh_consumed, is_filled, temperature (15-minute grid)
    resolution: pd.Timedelta  # the file's own reading interval (at least 15 minutes): scores use it
    notes: list[str] = field(default_factory=list)


def numbers(values: pd.Series) -> pd.Series:
    """Numbers, also as many exports write them: a decimal comma (``1,5``, ``1.234,5``) or a
    thousands comma (``1,234.5``). A comma before exactly three digits is ambiguous; it is read as a
    thousands separator unless another value in the column has a decimal comma."""
    parsed = pd.to_numeric(values, errors="coerce")
    if pd.api.types.is_numeric_dtype(values) or parsed.notna().sum() >= 0.9 * values.notna().sum():
        return parsed
    text = values.astype(str).str.strip()
    if text.str.contains(r",(?:\d{1,2}|\d{4,})$").any():  # 1,5 or 0,1234: a decimal comma
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce").where(values.notna())


def guess_type(name: str) -> str | None:
    """The model's building type for a meter name, or ``None`` when the name does not say."""
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name)).lower())
    for prefix, kind in TYPE_WORDS:
        if any(word.startswith(prefix) for word in words):
            return kind
    return None


def _parses_as_time(values: pd.Series) -> bool:
    sample = values.dropna().astype(str).head(50)
    if sample.empty or numbers(sample).notna().all():
        return False
    return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() >= 0.9


def guess_columns(raw: pd.DataFrame) -> dict:
    """Best guesses for the column pickers: timestamp, numeric, temperature and building columns,
    and whether dates are written day first (25/03/2026)."""
    numeric = [c for c in raw.columns
               if raw[c].notna().any() and numbers(raw[c]).notna().sum() >= 0.9 * raw[c].notna().sum()]
    times = [c for c in raw.columns if c not in numeric and _parses_as_time(raw[c])]
    temperature = next((c for c in numeric if re.search(r"temp|°c|deg", str(c), re.IGNORECASE)), None)
    labels = [c for c in raw.columns if c not in numeric and c not in times and 1 < raw[c].nunique() <= 200]
    dayfirst = False
    if times:
        first = raw[times[0]].dropna().astype(str).str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
        dayfirst = bool(pd.to_numeric(first[0], errors="coerce").gt(12).any())
    return {"timestamp": times[0] if times else None, "numeric": numeric, "temperature": temperature,
            "building": labels[0] if labels else None, "meters": [c for c in numeric if c != temperature],
            "dayfirst": dayfirst}


def _timestamps(raw: pd.Series, dayfirst: bool, notes: list[str]) -> pd.Series:
    try:
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (ValueError, TypeError):  # mixed UTC offsets
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst, format="mixed", utc=True)
        notes.append("Timestamps mix UTC offsets; they were converted to UTC.")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_localize(None)  # keep the local wall-clock time
        notes.append("Timezone dropped; local clock time kept.")
    if ts.isna().mean() > 0.5:
        raise ValueError("Most timestamps could not be read; check the timestamp column and the day-first setting.")
    if ts.isna().any():
        notes.append(f"{int(ts.isna().sum()):,} rows with unreadable timestamps were skipped.")
    return ts


def _to_slots(values: pd.Series, unit: str, native: pd.Timedelta, building: str, notes: list[str]) -> pd.Series:
    """One building's readings as kWh per 15-minute slot (NaN where a slot is not fully covered)."""
    if unit == REGISTER:
        step = values.index.to_series().diff().shift(-1)
        values = (values.shift(-1) - values).where(step == native)  # energy from each reading to the next
    elif unit == KW:
        values = values * (native / pd.Timedelta("1h"))
    negative = int((values < 0).sum())
    if negative:
        notes.append(f"{building}: {negative:,} negative values (e.g. meter resets) were dropped.")
    values = values[values >= 0]
    if native <= FREQ:
        grouped = values.groupby(values.index.floor(FREQ))
        return grouped.sum().where(grouped.count() == FREQ // native)
    parts = native // FREQ  # spread each coarse reading evenly over its 15-minute slots
    start = values.index.floor(FREQ)
    spread = pd.concat([pd.Series(values.to_numpy() / parts, index=start + k * FREQ) for k in range(parts)])
    return spread.groupby(level=0).sum()


def prepare(raw: pd.DataFrame, timestamp: str, meters: list[str], unit: str = KWH, building: str | None = None,
            temperature: str | None = None, dayfirst: bool = False, keep: list[str] | None = None) -> MeterData:
    """``raw`` as uploaded, onto complete months of a 15-minute grid. With ``building``, ``meters``
    is the single value column and each label in ``building`` is one meter (only those in ``keep``,
    if given); without it, each column in ``meters`` is one meter."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    if not meters:
        raise ValueError("Pick at least one meter column.")
    if building and len(meters) != 1:
        raise ValueError("With a building column, pick exactly one value column.")
    notes: list[str] = []
    ts = _timestamps(raw[timestamp], dayfirst, notes)
    if building:
        long = pd.DataFrame({"timestamp": ts, "building": raw[building].astype(str).str.strip(),
                             "value": numbers(raw[meters[0]])})
    else:
        long = pd.concat([pd.DataFrame({"timestamp": ts, "building": str(m), "value": numbers(raw[m])})
                          for m in meters], ignore_index=True)
    if building and keep is not None:
        long = long[long["building"].isin(keep)]
    long = long.dropna(subset=["timestamp", "value"])

    weather = None
    if temperature:
        temp = pd.DataFrame({"timestamp": ts, "t": numbers(raw[temperature])}).dropna()
        weather = temp.groupby(temp["timestamp"].dt.floor(FREQ))["t"].mean()

    frames, resolutions = [], []
    for name, rows in long.groupby("building", sort=False):
        duplicates = int(rows["timestamp"].duplicated().sum())
        if duplicates:
            notes.append(f"{name}: {duplicates:,} repeated timestamps (e.g. a clock change); the first reading was kept.")
        values = rows.groupby("timestamp")["value"].first().sort_index()
        if len(values) < 2:
            notes.append(f"{name}: fewer than two readings; skipped.")
            continue
        native = values.index.to_series().diff().median()
        if not ((native <= FREQ and FREQ % native == pd.Timedelta(0)) or (native > FREQ and native % FREQ == pd.Timedelta(0))):
            raise ValueError(f"{name}: readings are every {native}; the interval must divide 15 minutes "
                             "(1, 3, 5, 15 min) or be a multiple of it (30 min, 1 h, ...).")
        kwh = _to_slots(values, unit, native, name, notes)
        grid = pd.date_range(kwh.index.min().to_period("M").start_time, kwh.index.max().to_period("M").end_time.floor(FREQ),
                             freq=FREQ)
        kwh = kwh.reindex(grid)
        coverage = kwh.notna().groupby(grid.to_period("M")).mean()
        kept = coverage[coverage >= MIN_COVERAGE].index
        dropped = coverage[coverage < MIN_COVERAGE]
        if len(dropped):
            notes.append(f"{name}: {len(dropped)} month(s) left out with under {MIN_COVERAGE:.0%} of readings "
                         f"({', '.join(f'{p:%b %Y} {c:.0%}' for p, c in dropped.items())}); their totals would not be a real bill.")
        if not len(kept):
            continue
        in_kept = grid.to_period("M").isin(kept)
        kwh = kwh[in_kept]
        filled = kwh.isna()
        if filled.any():
            notes.append(f"{name}: {int(filled.sum()):,} missing 15-minute readings in kept months filled by "
                         "interpolation (never scored or trained on).")
        frames.append(pd.DataFrame({"timestamp": kwh.index, "building": name,
                                    "kWh_consumed": kwh.interpolate(limit_direction="both").to_numpy(),
                                    "is_filled": filled.to_numpy()}))
        resolutions.append(max(native, FREQ))
    if not frames:
        raise ValueError(f"No meter has a month with at least {MIN_COVERAGE:.0%} of its readings.")
    readings = pd.concat(frames, ignore_index=True)
    if weather is not None:
        grid = pd.DatetimeIndex(readings["timestamp"].unique()).sort_values()
        weather = weather.reindex(weather.index.union(grid)).interpolate(limit_direction="both").reindex(grid)
        readings["temperature"] = readings["timestamp"].map(weather)
    else:
        readings["temperature"] = np.nan
        notes.append(f"No temperature column: the weather is held at {NEUTRAL_TEMPERATURE:.0f} °C, so cooling "
                     "load is not modelled.")
    readings["temperature"] = readings["temperature"].fillna(NEUTRAL_TEMPERATURE)
    return MeterData(readings, max(resolutions), notes)


def hide_between(timestamps: pd.Series, start, end) -> np.ndarray:
    """Every reading from ``start`` up to (not including) ``end``."""
    return ((timestamps >= pd.Timestamp(start)) & (timestamps < pd.Timestamp(end))).to_numpy()


def hide_random_days(timestamps: pd.Series, share: float, seed: int = ml_model.SEED) -> np.ndarray:
    """A random ``share`` of whole days."""
    days = timestamps.dt.normalize()
    unique = np.sort(days.unique())
    chosen = np.random.default_rng(seed).choice(unique, size=round(share * len(unique)), replace=False)
    return days.isin(chosen).to_numpy()


def hide_hours(timestamps: pd.Series, start_hour: int, end_hour: int) -> np.ndarray:
    """The same hours of every day, e.g. (16, 20) for every 4-8 PM."""
    hour = timestamps.dt.hour
    return ((hour >= start_hour) & (hour < end_hour)).to_numpy()


def rebuild(data: MeterData, types: dict[str, str], building: str, hidden: np.ndarray,
            fallback: ShapeFn | None = None) -> tuple[pd.DataFrame, str]:
    """Hide ``hidden`` (one flag per reading of ``building``) and reconstruct it. Returns
    (per-interval ``actual``, ``predicted``, ``flat``, ``hidden``, ``filled`` indexed by timestamp,
    what the model was trained on)."""
    readings = data.readings.assign(building_type=data.readings["building"].map(types))
    own = (readings["building"] == building).to_numpy()
    hidden = np.asarray(hidden, dtype=bool)
    if hidden.shape != (own.sum(),):
        raise ValueError(f"expected {own.sum()} hidden flags for {building}, got {hidden.shape}")
    hide = np.zeros(len(readings), dtype=bool)
    hide[own] = hidden
    month = readings["timestamp"].dt.to_period("M").astype(str).rename("month")
    bills = (readings.groupby(["building", "building_type", month])["kWh_consumed"].sum()
             .rename("monthly_kWh").reset_index())
    reference = readings.assign(kWh_consumed=readings["kWh_consumed"].mask(hide | readings["is_filled"].to_numpy()))
    reference = reference[["timestamp", "building", "building_type", "temperature", "kWh_consumed"]]
    if reference["kWh_consumed"].notna().any():
        model = ml_model.train(reference, bills)
        shape, trained_on = ml_model.shape_fn(model, reference, bills), "the readings left visible in your file"
    elif fallback is not None:
        shape, trained_on = fallback, "the campus buildings (nothing in your file was left visible)"
    else:
        raise ValueError("Every reading is hidden and there is no fallback model.")

    months = bills.loc[bills["building"] == building, "month"]
    rebuilt = {name: pd.concat([reconstruct_building_month(reference, bills, building, m, fn) for m in months])
               .set_index("timestamp")["kWh_consumed"] for name, fn in (("predicted", shape), ("flat", flat_shape))}
    truth = readings[own].set_index("timestamp")
    series = pd.DataFrame({"actual": truth["kWh_consumed"], **rebuilt, "hidden": hidden,
                           "filled": truth["is_filled"].to_numpy()})
    return series, trained_on


def score(series: pd.DataFrame, resolution: pd.Timedelta = FREQ) -> pd.DataFrame:
    """Errors on the hidden real readings, at ``resolution`` (only periods hidden throughout), for the
    model and the flat split. Rows ``model`` and ``flat_bill_split``."""
    scored = series["hidden"] & ~series["filled"]
    by = series.index.floor(resolution)
    periods = series.assign(scored=scored).groupby(by).agg(
        actual=("actual", "sum"), predicted=("predicted", "sum"), flat=("flat", "sum"), scored=("scored", "all"))
    periods = periods[periods["scored"]]
    if periods.empty:
        raise ValueError("Nothing hidden to score: hide some readings first.")
    peak = (periods.index.hour >= PEAK_HOURS[0]) & (periods.index.hour < PEAK_HOURS[1])
    rows = {}
    for name, column in (("model", "predicted"), ("flat_bill_split", "flat")):
        result = score_prediction(periods["actual"], periods[column], periods.index.to_series())
        peak_actual = periods.loc[peak, "actual"].sum()
        rows[name] = {**result, "scored_periods": len(periods),
                      "peak_window_error_pct": 100 * result["peak_window_error_kwh"] / peak_actual if peak_actual else np.nan}
    return pd.DataFrame(rows).T
