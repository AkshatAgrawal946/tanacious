"""Validated, versioned forecasting experiment configuration.

Defaults reproduce the initial benchmark; ``configs/forecast_benchmark.json`` is the checked-in copy.
The splits are experimental choices for the synthetic 2026 data, not production defaults.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

CONFIG_VERSION = 1
COVARIATE_SCENARIOS = ("calendar", "calendar+oracle_temperature")


@dataclass(frozen=True)
class ForecastExperimentConfig:
    config_version: int = CONFIG_VERSION
    buildings: tuple[str, ...] = ("B3_Academic", "B5_Lab")
    context_steps: int = 1344  # 14 days of 15-minute history
    horizon_steps: int = 96  # 24 hours ahead
    # Half-open [start, end) date ranges.
    train: tuple[str, str] = ("2026-01-01", "2026-08-01")
    validation: tuple[str, str] = ("2026-08-01", "2026-09-01")
    test: tuple[str, str] = ("2026-09-01", "2026-10-01")
    origin_stride_steps: int = 96  # one forecast origin per day, at midnight
    # An origin is skipped for every model if more than this share of its target window is missing.
    max_missing_target_fraction: float = 0.5
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    # "calendar": only covariates known in advance (time of day, weekday, academic breaks).
    # "calendar+oracle_temperature": also the *measured* future temperature, an explicitly
    # declared perfect-weather-forecast scenario that is never the default.
    covariate_scenario: str = "calendar"
    seed: int = 2026
    chronos_model_id: str = "amazon/chronos-2"
    device: str = "cpu"
    inference_timeout_s: float = 300.0
    history_lengths_days: tuple[int, ...] = (1, 2, 3, 7, 14)
    # Promotion gate (an engineering rule, not a performance claim).
    min_relative_wape_improvement: float = 0.10
    max_peak_energy_error_regression_pct_points: float = 1.0
    max_peak_timing_regression_min: float = 30.0
    max_latency_s_per_forecast: float = 5.0

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise ValueError(f"config_version {self.config_version} is not supported (expected {CONFIG_VERSION})")
        if self.context_steps < self.horizon_steps or self.horizon_steps <= 0:
            raise ValueError("context_steps must be >= horizon_steps > 0")
        if self.covariate_scenario not in COVARIATE_SCENARIOS:
            raise ValueError(f"covariate_scenario must be one of {COVARIATE_SCENARIOS}")
        if not all(0 < q < 1 for q in self.quantile_levels) or 0.5 not in self.quantile_levels:
            raise ValueError("quantile_levels must lie in (0, 1) and include 0.5")
        if not 0 <= self.max_missing_target_fraction < 1:
            raise ValueError("max_missing_target_fraction must be in [0, 1)")
        bounds = [pd.Timestamp(d) for split in (self.train, self.validation, self.test) for d in split]
        if any(a > b for a, b in zip(bounds, bounds[1:])) or any(s >= e for s, e in zip(bounds[::2], bounds[1::2])):
            raise ValueError("train, validation and test must be non-empty, ordered and non-overlapping")

    @classmethod
    def from_json(cls, path: str | Path) -> "ForecastExperimentConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    def split(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, end = getattr(self, name)
        return pd.Timestamp(start), pd.Timestamp(end)
