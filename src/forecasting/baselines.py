"""Simple forecasting baselines. Every model sees exactly the same ``ForecastRequest``.

* ``SeasonalNaive(96)``: each interval repeats the same time on the previous day.
* ``SeasonalNaive(672)``: same weekday and time in the previous week.
  A missing lagged reading falls back to the same time one more period earlier, then to the
  history median, so dropouts in the history do not produce gaps.
* ``RandomForestForecaster``: a chronological, direct multi-step Random Forest. Its features come
  only from the request (lagged history, history level, known-in-advance covariates); it is trained
  on origins inside the training split only, and never sees bills or future measurements.
"""
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecasting.interfaces import FREQ, ForecastRequest, ForecastResult, validate_result


def _seasonal_values(request: ForecastRequest, period: int) -> np.ndarray:
    history = request.history
    fallback = float(np.nanmedian(history.to_numpy())) if history.notna().any() else 0.0
    values = np.empty(request.horizon)
    for k, t in enumerate(request.forecast_index):
        lag = t - period * FREQ
        while lag >= request.origin:  # horizon longer than the period: step back to the history
            lag -= period * FREQ
        value = np.nan
        while lag >= history.index[0] and np.isnan(value):
            value = history.get(lag, np.nan)
            lag -= period * FREQ
        values[k] = fallback if np.isnan(value) else value
    return values


class SeasonalNaive:
    def __init__(self, period_steps: int, name: str):
        self.period_steps, self.name, self.version = period_steps, name, f"seasonal-{period_steps}-v1"

    def predict(self, requests: Sequence[ForecastRequest]) -> list[ForecastResult]:
        results = []
        for request in requests:
            start = time.perf_counter()
            mean = _seasonal_values(request, self.period_steps)
            results.append(validate_result(request, ForecastResult(
                building=request.building, origin=request.origin, timestamps=request.forecast_index, mean=mean,
                model_name=self.name, model_version=self.version, latency_s=time.perf_counter() - start)))
        return results


def previous_day() -> SeasonalNaive:
    return SeasonalNaive(96, "seasonal_previous_day")


def previous_week() -> SeasonalNaive:
    return SeasonalNaive(672, "seasonal_previous_week")


class RandomForestForecaster:
    """Direct multi-step Random Forest on scaled targets (pooled over buildings, no building ID)."""

    name = "random_forest_forecast"
    version = "rf-forecast-v1"

    def __init__(self, seed: int = 2026, n_estimators: int = 200, min_samples_leaf: int = 5):
        self.seed = seed
        self.model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                           max_features=0.6, n_jobs=-1, random_state=seed)
        self.feature_names: list[str] | None = None
        self.training_cutoff: pd.Timestamp | None = None

    @staticmethod
    def _features(request: ForecastRequest) -> tuple[pd.DataFrame, float]:
        history = request.history
        scale = float(np.nanmean(history.to_numpy())) if history.notna().any() else 1.0
        scale = scale if scale > 0 else 1.0
        recent = history.iloc[-4:].dropna()
        frame = pd.DataFrame({
            "step": np.arange(request.horizon),
            "lag_day": _seasonal_values(request, 96) / scale,
            "lag_week": _seasonal_values(request, 672) / scale,
            "last_level": (recent.mean() if len(recent) else scale) / scale,
            "history_missing_share": history.isna().mean(),
        })
        if request.future_covariates is not None:
            for column in request.future_covariates.columns:
                frame[column] = request.future_covariates[column].to_numpy()
        return frame, scale

    def fit(self, requests: Sequence[ForecastRequest], targets: Sequence[pd.Series]) -> "RandomForestForecaster":
        frames, ys = [], []
        for request, target in zip(requests, targets):
            if not target.index.equals(request.forecast_index):
                raise ValueError("training targets must be aligned to their request")
            features, scale = self._features(request)
            keep = target.notna().to_numpy()
            frames.append(features[keep])
            ys.append(target.to_numpy()[keep] / scale)
        if not frames:
            raise ValueError("no training requests")
        features = pd.concat(frames, ignore_index=True)
        self.feature_names = list(features.columns)
        self.model.fit(features, np.concatenate(ys))
        self.training_cutoff = max(r.forecast_index[-1] for r in requests)
        return self

    def predict(self, requests: Sequence[ForecastRequest]) -> list[ForecastResult]:
        if self.feature_names is None:
            raise RuntimeError("fit the Random Forest forecaster before predicting")
        results = []
        for request in requests:
            if self.training_cutoff is not None and request.origin <= self.training_cutoff:
                raise ValueError(f"forecast origin {request.origin} is inside the training period (ends {self.training_cutoff})")
            start = time.perf_counter()
            features, scale = self._features(request)
            mean = self.model.predict(features[self.feature_names]) * scale
            results.append(validate_result(request, ForecastResult(
                building=request.building, origin=request.origin, timestamps=request.forecast_index, mean=mean,
                model_name=self.name, model_version=self.version, latency_s=time.perf_counter() - start,
                metadata={"training_cutoff": str(self.training_cutoff)})))
        return results
