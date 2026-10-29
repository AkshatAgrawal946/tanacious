"""Model routing: reconstruction for bill-only history, forecasting where genuine history exists.

* ``route(...)`` decides per building and forecast origin. A building is forecast only if it has at
  least ``min_history_steps`` *measured* readings in the context window before the origin. The
  threshold comes from the last benchmark's validated history-length experiment
  (``summary.json``); without one, a conservative 14-day default is used and the reason is logged.
* ``forecaster_for(...)`` returns the requested model, or the benchmark's default model (the
  promoted candidate, otherwise the strongest simple baseline). If the chosen model cannot run, it
  falls back to the previous-week seasonal baseline and reports the fallback; it is never silent.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import provenance
from src.forecasting.baselines import previous_day, previous_week
from src.forecasting.chronos_adapter import ZERO_SHOT_ID, ChronosForecaster
from src.forecasting.interfaces import FREQ, ForecasterUnavailable, ForecastRequest, ForecastResult

log = logging.getLogger(__name__)
SUMMARY_PATH = Path("data/processed/forecast_benchmark/summary.json")
DEFAULT_MIN_HISTORY_STEPS = 1344
FALLBACK_MODEL = "seasonal_previous_week"


@dataclass
class Route:
    building: str
    mode: str  # "forecast" or "reconstruct"
    measured_steps: int
    min_history_steps: int
    reason: str


@dataclass
class Selection:
    results: list[ForecastResult]
    requested: str
    used: str
    fallback_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def benchmark_summary(path: Path = SUMMARY_PATH) -> dict | None:
    return provenance.read_json(path)


def min_history_steps(summary: dict | None) -> tuple[int, str]:
    threshold = (summary or {}).get("history_threshold", {})
    if threshold.get("status") == "measured":
        return int(threshold["min_history_steps"]), f"validated on {threshold['model']} (benchmark {summary['created_at']})"
    log.info("no validated history threshold; using the %d-step default", DEFAULT_MIN_HISTORY_STEPS)
    return DEFAULT_MIN_HISTORY_STEPS, "no validated benchmark; conservative 14-day default"


def route(series: pd.Series | None, building: str, origin: pd.Timestamp, context_steps: int,
          summary: dict | None = None) -> Route:
    needed, why = min_history_steps(summary)
    if series is None:
        return Route(building, "reconstruct", 0, needed, "no interval meter: bills only")
    window = series[(series.index >= origin - context_steps * FREQ) & (series.index < origin)]
    measured = int(window.notna().sum())
    if measured >= needed:
        return Route(building, "forecast", measured, needed, f"{measured} measured readings before the origin ({why})")
    return Route(building, "reconstruct", measured, needed, f"only {measured} measured readings, {needed} needed ({why})")


def build_forecaster(name: str, device: str = "cpu", train_forest=None):
    """Instantiate a forecaster by name. ``train_forest`` is a callable returning a fitted forest."""
    if name == "seasonal_previous_day":
        return previous_day()
    if name == "seasonal_previous_week":
        return previous_week()
    if name == "random_forest_forecast":
        if train_forest is None:
            raise ForecasterUnavailable("the Random Forest forecaster needs training data")
        return train_forest()
    if name == "chronos2_zero_shot":
        return ChronosForecaster(ZERO_SHOT_ID, device=device)
    raise ForecasterUnavailable(f"unknown forecasting model {name!r}")


def forecast_with_fallback(requests: list[ForecastRequest], requested: str | None = None, summary: dict | None = None,
                           device: str = "cpu", train_forest=None, cache: dict | None = None) -> Selection:
    """Forecast with the requested (or benchmark default) model, falling back explicitly."""
    chosen = requested or (summary or {}).get("default_model") or FALLBACK_MODEL
    cache = cache if cache is not None else {}
    try:
        model = cache.get(chosen) or build_forecaster(chosen, device, train_forest)
        cache[chosen] = model
        return Selection(model.predict(requests), requested=chosen, used=chosen)
    except (ForecasterUnavailable, ValueError, RuntimeError) as error:
        if chosen == FALLBACK_MODEL:
            raise
        log.warning("forecaster %s unavailable (%s); falling back to %s", chosen, error, FALLBACK_MODEL)
        fallback = previous_week()
        return Selection(fallback.predict(requests), requested=chosen, used=FALLBACK_MODEL, fallback_reason=str(error))

