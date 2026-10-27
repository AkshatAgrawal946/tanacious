"""Accuracy check for the Random Forest reconstruction.

* Bill-only buildings (hostels, library): reconstructed from every real reading (smart meters and
  pilot meters) and scored against the hidden ground truth outside the pilot-meter weeks.
* Metered buildings (academic, lab): each is held out in turn, reconstructed from the other
  buildings plus its own bill, and scored against its real readings. This is the "new building
  type with no meter" test.
"""
import pandas as pd

from src import ml_model
from src.metrics import score_prediction
from src.reconstruct import ShapeFn, reconstruct_building_month


def _reconstruct(reference: pd.DataFrame, bills: pd.DataFrame, building: str, shape: ShapeFn) -> pd.Series:
    months = sorted(bills.loc[bills["building"] == building, "month"].astype(str))
    frames = [reconstruct_building_month(reference, bills, building, month, shape) for month in months]
    return pd.concat(frames).set_index("timestamp")["kWh_consumed"]


def _held_out(interval: pd.DataFrame, bills: pd.DataFrame, building: str) -> pd.Series:
    """Reconstruct a metered building from the *other* buildings' readings only."""
    reference = interval[interval["building"] != building]
    return _reconstruct(reference, bills, building, ml_model.shape_fn(ml_model.train(reference, bills), reference, bills))


def _weekday_peak_hour(values: pd.Series) -> int:
    weekday = values[values.index.day_of_week < 5]
    return int(weekday.groupby(weekday.index.hour).mean().idxmax())


def evaluate_model(interval: pd.DataFrame, bills: pd.DataFrame, hidden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-interval predictions next to the truth, per-building scores)."""
    bills = bills.assign(month=bills["month"].astype(str))
    forest = ml_model.shape_fn(ml_model.train(interval, bills), interval, bills)
    frames = []
    for building, building_type in bills.groupby("building")["building_type"].first().items():
        readings = interval.loc[interval["building"] == building, "timestamp"]
        if building in set(hidden["building"]):
            actual = hidden[hidden["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted = _reconstruct(interval, bills, building, forest)
            evaluated_against = "hidden ground truth"
            # Pilot-meter weeks are real readings the model keeps as-is, so they are not scored.
            is_pilot = predicted.index.isin(readings)
        else:
            actual = interval[interval["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted, evaluated_against, is_pilot = _held_out(interval, bills, building), "held-out meter", False
        frame = pd.DataFrame({"predicted": predicted, "actual": actual.reindex(predicted.index), "is_pilot": is_pilot})
        frame = frame.rename_axis("timestamp").reset_index()
        frames.append(frame.assign(building=building, building_type=building_type, evaluated_against=evaluated_against))
    series = pd.concat(frames, ignore_index=True)[
        ["timestamp", "building", "building_type", "evaluated_against", "is_pilot", "actual", "predicted"]]
    return series, score_series(series)


def score_series(series: pd.DataFrame) -> pd.DataFrame:
    """Per-building scores for any slice of ``evaluate_model`` output (e.g. some months).
    Pilot-meter intervals are left out: the model keeps those real readings as-is."""
    rows = []
    for (building, building_type, evaluated_against), group in series[~series["is_pilot"]].groupby(
            ["building", "building_type", "evaluated_against"]):
        group = group.dropna(subset=["actual"])
        peak = (group["timestamp"].dt.hour >= 16) & (group["timestamp"].dt.hour < 20)
        scores = score_prediction(group["actual"], group["predicted"], group["timestamp"])
        rows.append({
            "building": building, "building_type": building_type, "evaluated_against": evaluated_against,
            "mae_kwh": scores["mae_kwh"], "mape_pct": scores["mape_pct"],
            "peak_window_error_pct": 100 * scores["peak_window_error_kwh"] / group.loc[peak, "actual"].sum(),
            "true_peak_hour": _weekday_peak_hour(group.set_index("timestamp")["actual"]),
            "predicted_peak_hour": _weekday_peak_hour(group.set_index("timestamp")["predicted"]),
        })
    return pd.DataFrame(rows)
