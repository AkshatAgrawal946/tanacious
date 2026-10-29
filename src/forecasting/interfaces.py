"""Forecasting contracts shared by every model.

A ``ForecastRequest`` carries only what is legitimately known at its origin: measured history that
ends strictly before the origin, and future covariates that are known in advance. It refuses to be
built otherwise, so leakage is caught where requests are made, not after scoring.
"""
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

FREQ = pd.Timedelta("15min")


class ForecasterUnavailable(RuntimeError):
    """The model cannot produce forecasts (missing dependency, checkpoint, credentials or service)."""


class ForecastTimeout(ForecasterUnavailable):
    """Inference took longer than its configured limit."""


@dataclass(frozen=True)
class ForecastRequest:
    building: str
    origin: pd.Timestamp  # first forecast timestamp; history ends one step earlier
    history: pd.Series  # kWh on a contiguous 15-minute index; NaN = missing reading
    horizon: int
    past_covariates: pd.DataFrame | None = None  # aligned to history.index
    future_covariates: pd.DataFrame | None = None  # index == forecast_index; known-in-advance only

    def __post_init__(self):
        index = self.history.index
        if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            raise ValueError("history needs a non-empty DatetimeIndex")
        if index[-1] != self.origin - FREQ:
            raise ValueError(f"history must end one step before the origin {self.origin}, not at {index[-1]} (leakage guard)")
        if len(index) > 1 and not (index.to_series().diff().iloc[1:] == FREQ).all():
            raise ValueError("history must be on a contiguous 15-minute grid (use NaN for missing readings)")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.past_covariates is not None and not self.past_covariates.index.equals(index):
            raise ValueError("past_covariates must be aligned to the history index")
        if self.future_covariates is not None:
            if not self.future_covariates.index.equals(self.forecast_index):
                raise ValueError("future_covariates must cover exactly the forecast timestamps")
            if self.past_covariates is None or not set(self.future_covariates.columns) <= set(self.past_covariates.columns):
                raise ValueError("every future covariate needs its past values in past_covariates")
            if self.future_covariates.isna().any().any():
                raise ValueError("future covariates must be fully known")

    @property
    def forecast_index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin, periods=self.horizon, freq=FREQ)


@dataclass
class ForecastResult:
    building: str
    origin: pd.Timestamp
    timestamps: pd.DatetimeIndex
    mean: np.ndarray  # point forecast (median for probabilistic models)
    model_name: str
    model_version: str
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    latency_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"timestamp": self.timestamps, "building": self.building, "origin": self.origin,
                              "model": self.model_name, "model_version": self.model_version, "predicted": self.mean})
        for level, values in sorted(self.quantiles.items()):
            frame[f"q{round(level * 100):02d}"] = values
        return frame


class Forecaster(Protocol):
    name: str
    version: str

    def predict(self, requests: Sequence[ForecastRequest]) -> list[ForecastResult]: ...


def validate_result(request: ForecastRequest, result: ForecastResult) -> ForecastResult:
    """Check a model's output against its request: timestamps, shape, finiteness and quantile
    order. Negative energy is clipped to zero (recorded in metadata)."""
    if result.building != request.building or result.origin != request.origin:
        raise ValueError(f"{result.model_name} answered a different request ({result.building} @ {result.origin})")
    if not result.timestamps.equals(request.forecast_index):
        raise ValueError(f"{result.model_name} returned misaligned timestamps for {request.building} @ {request.origin}")
    for label, values in [("mean", result.mean), *[(f"q{q}", v) for q, v in result.quantiles.items()]]:
        values = np.asarray(values, dtype=float)
        if values.shape != (request.horizon,):
            raise ValueError(f"{result.model_name} {label} has shape {values.shape}, expected ({request.horizon},)")
        if not np.isfinite(values).all():
            raise ValueError(f"{result.model_name} {label} contains NaN or infinite values")
    clipped = int((np.asarray(result.mean) < 0).sum())
    result.mean = np.clip(np.asarray(result.mean, dtype=float), 0, None)
    result.quantiles = {q: np.clip(np.asarray(v, dtype=float), 0, None) for q, v in result.quantiles.items()}
    levels = sorted(result.quantiles)
    for low, high in zip(levels, levels[1:]):
        if (result.quantiles[low] > result.quantiles[high] + 1e-9).any():
            raise ValueError(f"{result.model_name} quantiles cross ({low} > {high})")
    if clipped:
        result.metadata["negative_values_clipped"] = clipped
    return result
