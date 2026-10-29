"""Likely ranges for reconstructed load, calibrated on the model's own tested errors.

The pipeline scores the model by reconstructing load it never saw (a hidden true load, or a
building's own meter held out) and saves each prediction next to the truth (``evaluation_15min.csv``).
How far actual/predicted strayed in that test, per building and hour of day, is how far we expect
the app's imputed load to stray: a reconstructed 100 kW with ratios (0.90, 1.08) is "likely 90-108 kW".

Only imputed load gets a range. Real meter readings are measurements, and the bills are facts:
every month is matched to its bill, so the uncertainty is in *when* energy was used, not how much.
A range is the central 80% (P10-P90) of past errors, not a guarantee.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.9)
CAMPUS = "Campus"


@dataclass(frozen=True)
class Calibration:
    """Each table is indexed by building (plus ``CAMPUS``), with ``low`` and ``high`` ratio columns."""
    interval: pd.DataFrame     # (building, hour): one 15-minute interval
    average_day: pd.DataFrame  # (building, hour): one hour of a month's average day
    peak_energy: pd.DataFrame  # building: a month's 4-8 PM energy


def _spread(ratio: pd.Series, by) -> pd.DataFrame:
    """P10/P90 of ``ratio`` per group, widened to include 1 so a range always contains the value
    shown (a model that is biased one way, say always 3% high on a lab's evenings, still gets one)."""
    table = ratio.groupby(by).quantile(list(QUANTILES)).unstack()
    table.columns = ["low", "high"]
    return table.assign(low=table["low"].clip(upper=1), high=table["high"].clip(lower=1))


def calibrate(series: pd.DataFrame) -> Calibration:
    """Ratio spreads from ``evaluate_model`` output. Pilot-meter intervals are left out: the model
    keeps those real readings as-is, so they would make the ranges look tighter than they are."""
    scored = series[~series["is_pilot"]].dropna(subset=["actual"])[["timestamp", "building", "actual", "predicted"]]
    campus = scored.groupby("timestamp")[["actual", "predicted"]].sum().reset_index().assign(building=CAMPUS)
    rows = pd.concat([scored, campus], ignore_index=True)
    rows = rows[rows["predicted"] > 0]
    building, hour = rows["building"], rows["timestamp"].dt.hour.rename("hour")
    month = rows["timestamp"].dt.to_period("M").rename("month")

    interval = _spread(rows["actual"] / rows["predicted"], [building, hour])
    slots = rows.groupby([building, month, hour])[["actual", "predicted"]].mean()
    average_day = _spread(slots["actual"] / slots["predicted"], ["building", "hour"])
    peak = hour.between(16, 19)
    evenings = rows[peak].groupby([building[peak], month[peak]])[["actual", "predicted"]].sum()
    peak_energy = _spread(evenings["actual"] / evenings["predicted"], "building")
    return Calibration(interval, average_day, peak_energy)


def likely_range(table: pd.DataFrame | None, imputed, observed, building, hour=None) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) for ``observed + imputed`` with only the imputed part scaled by ``table``'s ratios.
    ``building`` and ``hour`` may be scalars or arrays; unknown keys (or no table) give NaN."""
    imputed, observed = np.asarray(imputed, dtype=float), np.asarray(observed, dtype=float)
    shape = np.broadcast(imputed, observed).shape
    if table is None:
        return np.full(shape, np.nan), np.full(shape, np.nan)
    building = np.broadcast_to(np.asarray(building), shape).ravel()
    if hour is None:
        ratios = table.reindex(building)
    else:
        hour = np.broadcast_to(np.asarray(hour), shape).ravel()
        ratios = table.reindex(pd.MultiIndex.from_arrays([building, hour]))
    low = ratios["low"].to_numpy().reshape(shape)
    high = ratios["high"].to_numpy().reshape(shape)
    return observed + imputed * low, observed + imputed * high
