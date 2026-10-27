"""Random Forest reconstruction model (ported from the ml-imputation branch).

The forest learns how a building's load *shape* responds to its type's daily schedule, the
weather and the calendar, from the smart-metered buildings only. For a bill-only building it
predicts that shape from the building's own schedule; the bill then fixes the absolute level
through the same reconciliation the baseline uses (``reconstruct_building_month``).

Target: each reading divided by the building's average interval energy over all its bills
(total billed kWh / number of intervals), so buildings of different size share one scale while
seasonal and vacation drops stay visible.

Train and save with ``python -m src.ml_model``.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.campus_calendar import is_break
from src.reconstruct import ShapeFn, _type_prior

SEED = 2026
MODEL_PATH = Path("models/random_forest.joblib")
SCHEDULE_FEATURES = [
    "Is_Weekend", "Is_Break", "Temperature", "Temperature_3h_Mean", "Cooling_Degrees",
    "Scheduled_Occupancy", "Occupancy_Prev_1h", "Occupancy_Next_1h", "Occupied_Cooling",
    # Whole-day descriptors of the building type. Without them, two types whose hourly schedules
    # match at different times of day look identical (the library at 6 PM equals the lab at 2 PM),
    # and the forest copies the lab's flatter peak onto the library, leaving a dip at its peak.
    "Night_Occupancy", "Evening_Share",
]
# With clock time the forest memorises when the *metered* buildings are busy (academic and lab,
# both afternoon peaks) and copies that onto hostels and the library. Without it, load has to be
# explained through each type's own schedule, which is what transfers to unmetered types.
FEATURE_SETS = {
    "schedule_only": SCHEDULE_FEATURES,
    "schedule_and_clock": ["Hour_Sin", "Hour_Cos"] + SCHEDULE_FEATURES,
}
DEFAULT_FEATURE_SET = "schedule_only"


def campus_temperature(interval: pd.DataFrame) -> pd.Series:
    """One campus-wide temperature reading per timestamp (all buildings share the weather)."""
    return interval.groupby("timestamp")["temperature"].mean().sort_index()


def build_features(timestamps: pd.DatetimeIndex, building_type: str, temperature: pd.Series) -> pd.DataFrame:
    """Model inputs for one building over ``timestamps``. Uses only what is known without a
    meter: the clock, the weather and the building type's timetable."""
    timestamps = pd.DatetimeIndex(timestamps)
    hour = np.asarray(timestamps.hour + timestamps.minute / 60, dtype=float)
    temp = temperature.reindex(timestamps).interpolate(limit_direction="both")
    temp = temp.fillna(temperature.mean()).to_numpy()
    # Buildings respond to the weather with thermal lag; a trailing mean is causal.
    temp_3h = pd.Series(temp).rolling(12, min_periods=1).mean().to_numpy()
    occupancy = _type_prior(building_type, hour)
    cooling = np.clip(temp_3h - 26.0, 0, None)
    day = _type_prior(building_type, np.arange(24, dtype=float))
    return pd.DataFrame({
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Is_Weekend": (timestamps.day_of_week >= 5).astype(int),
        "Is_Break": is_break(timestamps).astype(int),  # academic calendar, known in advance
        "Temperature": temp,
        "Temperature_3h_Mean": temp_3h,
        "Cooling_Degrees": cooling,
        # The timetable is known in advance, so looking an hour ahead in it is not leakage.
        "Scheduled_Occupancy": occupancy,
        "Occupancy_Prev_1h": _type_prior(building_type, (hour - 1) % 24),
        "Occupancy_Next_1h": _type_prior(building_type, (hour + 1) % 24),
        "Occupied_Cooling": occupancy * cooling,
        # Always-on level (3 AM) and how evening-heavy the type is (5-11 PM share of the day).
        "Night_Occupancy": day[3],
        "Evening_Share": day[17:23].sum() / day.sum(),
    })


def _training_rows(interval: pd.DataFrame, bills: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    temperature = campus_temperature(interval)
    observed = interval.dropna(subset=["kWh_consumed"])
    if observed.empty:
        raise ValueError("no observed 15-minute data to train on")
    frames, targets = [], []
    for building, rows in observed.groupby("building"):
        rows = rows.sort_values("timestamp")
        # Average interval energy over all of the building's bills. A per-month average would hide
        # the drop in an all-vacation month (June looks "normal for June") and the forest would
        # never learn that breaks are quiet, which matters in part-term, part-break months.
        own_bills = bills[bills["building"] == building]
        intervals = sum(pd.Period(str(m), freq="M").days_in_month * 96 for m in own_bills["month"])
        frames.append(build_features(pd.DatetimeIndex(rows["timestamp"]), rows["building_type"].iloc[0], temperature))
        targets.append(rows["kWh_consumed"].to_numpy() / (own_bills["monthly_kWh"].sum() / intervals))
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def train(interval: pd.DataFrame, bills: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET,
          seed: int = SEED) -> RandomForestRegressor:
    """Fit on every real smart-meter reading in ``interval``."""
    features, target = _training_rows(interval, bills)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.6,
                                  n_jobs=-1, random_state=seed)
    model.fit(features[FEATURE_SETS[feature_set]], target)
    return model


def shape_fn(model: RandomForestRegressor, interval: pd.DataFrame) -> ShapeFn:
    """Wrap a trained forest as a shape for ``reconstruct_building_month``."""
    temperature = campus_temperature(interval)

    def predict(timestamps: pd.DatetimeIndex, building_type: str) -> np.ndarray:
        features = build_features(timestamps, building_type, temperature)
        return np.clip(model.predict(features[list(model.feature_names_in_)]), 1e-6, None)

    return predict


def save(model: RandomForestRegressor, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: Path = MODEL_PATH) -> RandomForestRegressor:
    return joblib.load(path)


if __name__ == "__main__":
    from src.data import load_data

    interval, bills = load_data()
    path = save(train(interval, bills))
    print(f"Trained a Random Forest ({DEFAULT_FEATURE_SET}) on {interval['kWh_consumed'].notna().sum():,} readings; saved to {path}")
