"""Turn validated meter data into leak-free forecasting requests.

* Only measured readings are used as history and actuals; missing readings stay NaN (Chronos-2
  accepts NaN in its context). Reconstructed values are never used.
* Covariates are limited to what is known in advance (time of day, weekday, academic calendar).
  Building identifiers are never fed as numbers. Measured future temperature is only used in the
  explicitly declared "calendar+oracle_temperature" scenario.
* Chronos-2 normalises each series internally; the Random Forest baseline scales by the history
  mean of each request and inverts it on output (``baselines.py``).
"""
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
