"""Turn validated meter data into leak-free forecasting requests and Chronos-2 training inputs.

* Only measured readings are targets; missing readings stay NaN (Chronos-2 accepts NaN in its
  context and ignores NaN targets in training). Reconstructed values are never used as targets.
* Covariates are limited to what is known in advance (time of day, weekday, academic calendar).
  Building identifiers are never fed as numbers. Measured future temperature is only used in the
  explicitly declared "calendar+oracle_temperature" scenario.
* Training data is cut at the end of the training split, so no training window can contain a
  validation or test target. Validation windows may look back into the training months for context
  (history that would be known at the time), but their targets lie inside the validation split only.
* Chronos-2 normalises each series internally; the Random Forest baseline scales by the history
  mean of each request and inverts it on output (``baselines.py``).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.campus_calendar import is_break
from src.forecasting.interfaces import FREQ, ForecastRequest

CALENDAR_COVARIATES = ["hour", "day_of_week", "is_weekend", "is_break"]


def building_series(interval: pd.DataFrame, building: str) -> pd.Series:
    """A building's measured kWh on a contiguous 15-minute grid (NaN where nothing was measured)."""
    rows = interval[interval["building"] == building]
    if rows.empty:
        raise ValueError(f"no interval data for {building}")
    series = rows.set_index("timestamp")["kWh_consumed"].astype(float).sort_index()
    if series.index.duplicated().any():
        raise ValueError(f"duplicate timestamps for {building}")
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq=FREQ))


def calendar_covariates(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Covariates known in advance for any timestamp."""
    return pd.DataFrame({
        "hour": index.hour + index.minute / 60,
        "day_of_week": index.day_of_week,
        "is_weekend": (index.day_of_week >= 5).astype(int),
        "is_break": is_break(index).astype(int),
    }, index=index).astype(float)


def covariates(index: pd.DatetimeIndex, scenario: str, temperature: pd.Series | None = None) -> pd.DataFrame:
    frame = calendar_covariates(index)
    if scenario == "calendar+oracle_temperature":
        if temperature is None:
            raise ValueError("the oracle-temperature scenario needs a temperature series")
        frame["temperature"] = temperature.reindex(index).interpolate(limit_direction="both").to_numpy()
    return frame


def make_request(series: pd.Series, building: str, origin: pd.Timestamp, context_steps: int, horizon: int,
                 scenario: str = "calendar", temperature: pd.Series | None = None) -> ForecastRequest:
    """History of up to ``context_steps`` readings strictly before ``origin``."""
    origin = pd.Timestamp(origin)
    start = max(origin - context_steps * FREQ, series.index[0])
    history_index = pd.date_range(start, origin - FREQ, freq=FREQ)
    if len(history_index) == 0:
        raise ValueError(f"no history before {origin} for {building}")
    history = series.reindex(history_index)
    forecast_index = pd.date_range(origin, periods=horizon, freq=FREQ)
    return ForecastRequest(building=building, origin=origin, history=history, horizon=horizon,
                           past_covariates=covariates(history_index, scenario, temperature),
                           future_covariates=covariates(forecast_index, scenario, temperature))


def actuals(series: pd.Series, origin: pd.Timestamp, horizon: int) -> pd.Series:
    """Measured values over a forecast window (evaluation only)."""
    return series.reindex(pd.date_range(origin, periods=horizon, freq=FREQ))


def forecast_origins(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, horizon: int, stride: int,
                     context_steps: int, max_missing_target_fraction: float) -> list[pd.Timestamp]:
    """Origins whose whole target window lies in [start, end), with a full context window of the
    series before them and enough measured targets to score."""
    origins = []
    origin = pd.Timestamp(start).normalize()
    while origin + horizon * FREQ <= end:
        enough_history = origin - context_steps * FREQ >= series.index[0]
        target = actuals(series, origin, horizon)
        if enough_history and target.isna().mean() <= max_missing_target_fraction:
            origins.append(origin)
        origin += stride * FREQ
    return origins


def _segments(values: pd.Series, max_gap_steps: int, min_length: int) -> list[pd.Series]:
    """Split at runs of more than ``max_gap_steps`` missing readings; shorter gaps stay as NaN."""
    missing = values.isna().to_numpy()
    run_id = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
    run_length = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    long_gap = missing & (run_length > max_gap_steps)
    segment_id = np.cumsum(long_gap)
    pieces = []
    for _, piece in values[~long_gap].groupby(segment_id[~long_gap]):
        piece = piece.loc[piece.first_valid_index():piece.last_valid_index()] if piece.notna().any() else piece.iloc[:0]
        if len(piece) >= min_length:
            pieces.append(piece)
    return pieces


def chronos_training_inputs(series_by_building: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp,
                            horizon: int, scenario: str = "calendar", temperature: pd.Series | None = None,
                            max_gap_steps: int = 96) -> list[dict]:
    """Training series in Chronos-2's list-of-dicts format, cut to [start, end)."""
    inputs = []
    for building, series in series_by_building.items():
        window = series[(series.index >= start) & (series.index < end)]
        for piece in _segments(window, max_gap_steps, min_length=2 * horizon):
            cov = covariates(piece.index, scenario, temperature)
            inputs.append({"target": piece.to_numpy(dtype=float),
                           "past_covariates": {c: cov[c].to_numpy() for c in cov.columns},
                           "item_id": f"{building}:{piece.index[0]:%Y-%m-%dT%H:%M}", "start": piece.index[0]})
    return inputs


def chronos_validation_inputs(series_by_building: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp,
                              context_steps: int, horizon: int, stride: int, scenario: str = "calendar",
                              temperature: pd.Series | None = None, max_missing_target_fraction: float = 0.5) -> list[dict]:
    """One series per validation origin: ``context_steps`` of history followed by ``horizon``
    targets inside [start, end) (Chronos-2 validation scores the last ``horizon`` values)."""
    inputs = []
    for building, series in series_by_building.items():
        for origin in forecast_origins(series, start, end, horizon, stride, context_steps, max_missing_target_fraction):
            index = pd.date_range(origin - context_steps * FREQ, periods=context_steps + horizon, freq=FREQ)
            cov = covariates(index, scenario, temperature)
            inputs.append({"target": series.reindex(index).to_numpy(dtype=float),
                           "past_covariates": {c: cov[c].to_numpy() for c in cov.columns},
                           "item_id": f"{building}:{origin:%Y-%m-%dT%H:%M}", "start": index[0]})
    return inputs


def known_covariate_names(scenario: str) -> list[str]:
    return CALENDAR_COVARIATES + (["temperature"] if scenario == "calendar+oracle_temperature" else [])


def to_chronos(inputs: list[dict]) -> list[dict]:
    """Strip bookkeeping keys before handing inputs to Chronos-2."""
    return [{"target": item["target"], "past_covariates": item["past_covariates"]} for item in inputs]


def validate_training_inputs(inputs: list[dict], horizon: int, split_end: pd.Timestamp | None = None) -> list[str]:
    """Problems that must block a training job (empty data, non-finite or negative targets,
    covariates misaligned with targets, or windows that run past the split end)."""
    problems = []
    if not inputs:
        problems.append("no training series")
    for item in inputs:
        target = np.asarray(item["target"], dtype=float)
        name = item.get("item_id", "?")
        if len(target) < 2 * horizon:
            problems.append(f"{name}: {len(target)} steps is shorter than two horizons")
        if np.isinf(target).any() or (target[~np.isnan(target)] < 0).any():
            problems.append(f"{name}: targets must be finite and non-negative")
        if np.isnan(target).all():
            problems.append(f"{name}: no measured values")
        for key, values in item.get("past_covariates", {}).items():
            if len(values) != len(target) or not np.isfinite(np.asarray(values, dtype=float)).all():
                problems.append(f"{name}: covariate {key} is misaligned or not finite")
        if split_end is not None and "start" in item and pd.Timestamp(item["start"]) + len(target) * FREQ > split_end:
            problems.append(f"{name}: runs past the split end {split_end}")
    return problems


def export_jsonl(inputs: list[dict], path: str | Path, known_covariates: list[str]) -> Path:
    """Write series as JSON lines in a provider-neutral layout (item_id, start, freq, target,
    past_covariates, known_covariates). Missing readings are written as null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in inputs:
            target = [None if np.isnan(v) else round(float(v), 6) for v in item["target"]]
            handle.write(json.dumps({
                "item_id": item.get("item_id"), "start": str(item.get("start")), "freq": "15min", "target": target,
                "past_covariates": {k: [float(v) for v in vals] for k, vals in item["past_covariates"].items()},
                "known_covariates": known_covariates,
            }) + "\n")
    return path
