"""Interpretable baseline reconstruction with a hard billing constraint."""
from typing import Callable

import numpy as np
import pandas as pd


# Coarse relative hourly load by building type: what a facilities team knows about when each
# kind of building is busy. Academic blocks peak with afternoon classes, libraries and hostels
# in the evening, labs stay high all day. Unknown types get a flat prior.
TYPE_PRIORS = {
    "academic": [.45] * 7 + [.5, .65, .8, .9, .9, .85, .95, 1., 1., .85, .65, .55, .5, .45, .45, .45, .45],
    "library": [.4] * 7 + [.4, .5, .6, .65, .7, .7, .75, .8, .85, .9, .95, 1., 1., .95, .85, .7, .55],
    "hostel": [.7, .65, .65, .65, .65, .7, .75, .75, .6, .5, .5, .5,
               .55, .55, .55, .6, .7, .8, .9, 1., 1., .95, .9, .8],
    "lab": [.65] * 7 + [.7, .8, .9, .95, .95, .95, .95, 1., 1., .95, .9, .85, .8, .75, .72, .7, .68],
}


def _type_prior(building_type: str, hour: np.ndarray) -> np.ndarray:
    prior = TYPE_PRIORS.get(building_type, [1.0] * 24)
    return np.interp(hour, np.arange(25), prior + prior[:1])


def _type_modifier(building_type: str, reference_types: pd.Series, hour: np.ndarray) -> np.ndarray:
    """Re-time the reference buildings' profile to the target type's daily pattern.

    Used only when the target type has no smart meter: the ratio moves load towards the hours
    the target type is busy and away from the hours the metered references are busy."""
    reference = np.mean([_type_prior(kind, hour) for kind in reference_types], axis=0)
    return _type_prior(building_type, hour) / reference


def _reference_shape(interval: pd.DataFrame, timestamps: pd.DatetimeIndex, building_type: str) -> np.ndarray:
    observed = interval.dropna(subset=["kWh_consumed"]).copy()
    if observed.empty:
        raise ValueError("at least one observed smart-meter value is required")
    hour = np.asarray(timestamps.hour + timestamps.minute / 60, dtype=float)
    same_type = observed[observed["building_type"] == building_type]
    if same_type.empty:
        modifier = _type_modifier(building_type, observed.groupby("building")["building_type"].first(), hour)
    else:
        observed, modifier = same_type.copy(), np.ones(len(hour))
    observed["slot"] = observed["timestamp"].dt.dayofweek * 96 + observed["timestamp"].dt.hour * 4 + observed["timestamp"].dt.minute // 15
    profile = observed.groupby("slot")["kWh_consumed"].mean()
    slots = timestamps.dayofweek * 96 + timestamps.hour * 4 + timestamps.minute // 15
    generic = float(observed["kWh_consumed"].mean())
    shape = np.asarray([profile.get(slot, generic) for slot in slots], dtype=float)
    # Cooling is already in the metered reference profile, so no separate temperature uplift.
    return np.clip(shape * modifier, 1e-6, None)


def reconcile_monthly(shape: np.ndarray, monthly_kwh: float) -> np.ndarray:
    """Scale a positive shape so its interval values sum exactly to the known bill."""
    shape = np.asarray(shape, dtype=float)
    if monthly_kwh < 0 or not np.isfinite(monthly_kwh):
        raise ValueError("monthly_kwh must be a finite non-negative number")
    if np.any(shape < 0) or not np.isfinite(shape).all() or shape.sum() <= 0:
        raise ValueError("shape must be finite, non-negative, and have positive mass")
    reconciled = shape * (monthly_kwh / shape.sum())
    # Correct float drift on the final interval; this makes the accounting invariant explicit.
    reconciled[-1] += monthly_kwh - reconciled.sum()
    return reconciled


ShapeFn = Callable[[pd.DatetimeIndex, str], np.ndarray]


def reconstruct_building_month(interval: pd.DataFrame, bills: pd.DataFrame, building: str, month: str,
                               shape_fn: ShapeFn | None = None) -> pd.DataFrame:
    """Reconstruct one building-month. ``shape_fn(timestamps, building_type)`` gives the
    relative load shape; by default the reference-profile baseline. The bill and any real
    readings are applied the same way whichever model supplies the shape."""
    bill = bills[(bills["building"] == building) & (bills["month"].astype(str) == str(month))]
    if len(bill) != 1:
        raise ValueError(f"expected one bill for {building} in {month}")
    building_type = bill.iloc[0]["building_type"]
    monthly_kwh = float(bill.iloc[0]["monthly_kWh"])
    period = pd.Period(str(month), freq="M")
    timestamps = pd.date_range(period.start_time, period.end_time, freq="15min")
    if shape_fn is None:
        shape = _reference_shape(interval, timestamps, building_type)
    else:
        shape = np.clip(np.asarray(shape_fn(timestamps, building_type), dtype=float), 1e-6, None)
    result = pd.DataFrame({
        "timestamp": timestamps, "building": building, "building_type": building_type,
        "month": str(month), "kWh_consumed": reconcile_monthly(shape, monthly_kwh),
        "is_observed": False, "is_reconstructed": True,
    })
    # Preserve actual smart-meter readings. Only the missing intervals are estimated, and
    # their predicted shape is rescaled to the energy left after actual readings are used.
    observed = interval.loc[
        (interval["building"] == building)
        & (interval["timestamp"].dt.to_period("M").astype(str) == str(month)),
        ["timestamp", "kWh_consumed"],
    ].dropna(subset=["kWh_consumed"])
    if observed.empty:
        return result

    known = observed.set_index("timestamp")["kWh_consumed"]
    observed_mask = result["timestamp"].isin(known.index)
    observed_kwh = float(result.loc[observed_mask, "timestamp"].map(known).sum())
    remaining_kwh = monthly_kwh - observed_kwh
    if remaining_kwh < -0.01:
        raise ValueError(
            f"observed interval energy ({observed_kwh:.2f} kWh) exceeds the monthly bill "
            f"({monthly_kwh:.2f} kWh) for {building} in {month}"
        )
    if observed_mask.all():
        if abs(remaining_kwh) > 0.01:
            raise ValueError(f"all intervals are observed but do not reconcile to the monthly bill for {building} in {month}")
    else:
        result.loc[~observed_mask, "kWh_consumed"] = reconcile_monthly(
            result.loc[~observed_mask, "kWh_consumed"].to_numpy(), max(remaining_kwh, 0.0)
        )
    result.loc[observed_mask, "kWh_consumed"] = result.loc[observed_mask, "timestamp"].map(known)
    result.loc[observed_mask, "is_observed"] = True
    result.loc[observed_mask, "is_reconstructed"] = False
    return result


def reconstruct_all(interval: pd.DataFrame, bills: pd.DataFrame, shape_fn: ShapeFn | None = None) -> pd.DataFrame:
    frames = [reconstruct_building_month(interval, bills, row.building, str(row.month), shape_fn) for row in bills.itertuples()]
    return pd.concat(frames, ignore_index=True)
