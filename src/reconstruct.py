"""Bill-constrained reconstruction: turn a predicted load shape into 15-minute readings that
keep every real meter reading and add up exactly to each monthly bill."""
from typing import Callable

import numpy as np
import pandas as pd


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


ShapeFn = Callable[[pd.DatetimeIndex, str, str], np.ndarray]  # (timestamps, building, building_type)


def reconstruct_building_month(interval: pd.DataFrame, bills: pd.DataFrame, building: str, month: str,
                               shape_fn: ShapeFn) -> pd.DataFrame:
    """Reconstruct one building-month. ``shape_fn(timestamps, building, building_type)`` (the
    Random Forest, see ``ml_model.shape_fn``) gives the relative load shape; real readings are
    kept and the rest is scaled so the month matches its bill."""
    bill = bills[(bills["building"] == building) & (bills["month"].astype(str) == str(month))]
    if len(bill) != 1:
        raise ValueError(f"expected one bill for {building} in {month}")
    building_type = bill.iloc[0]["building_type"]
    monthly_kwh = float(bill.iloc[0]["monthly_kWh"])
    period = pd.Period(str(month), freq="M")
    timestamps = pd.date_range(period.start_time, period.end_time, freq="15min")
    shape = np.clip(np.asarray(shape_fn(timestamps, building, building_type), dtype=float), 1e-6, None)
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


def reconstruct_all(interval: pd.DataFrame, bills: pd.DataFrame, shape_fn: ShapeFn) -> pd.DataFrame:
    frames = [reconstruct_building_month(interval, bills, row.building, str(row.month), shape_fn) for row in bills.itertuples()]
    return pd.concat(frames, ignore_index=True)
