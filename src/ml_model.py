"""Random Forest reconstruction model.

The forest learns how a building's load *shape* responds to its type's daily schedule, the
weather and the academic calendar, from every real 15-minute reading (smart meters and pilot
meters). For a bill-only building it predicts that shape from what is known about the building;
the bill then fixes the absolute level (``reconstruct_building_month``).

Target: each reading divided by the building's average interval energy over all its bills
(total billed kWh / number of intervals), so buildings of different size share one scale while
seasonal and vacation drops stay visible.

Information availability: ``break_ratios`` and ``_training_rows`` read *every* bill of a building,
including months after the reading being modelled. That is legitimate for retrospective
reconstruction (completed months are billed), but those bills would not exist when predicting
the future, so this model must not be used for forecasting.

Train and save with ``python -m src.ml_model``.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from src import provenance
from src.campus_calendar import BREAKS, is_break
from src.reconstruct import ShapeFn

log = logging.getLogger(__name__)
SEED = 2026
MODEL_PATH = Path("models/random_forest.joblib")
# Bump when the features, target or hyperparameters change: cached models of another version are retrained.
MODEL_VERSION = "rf-reconstruction-2"
HYPERPARAMETERS = {"n_estimators": 100, "min_samples_leaf": 20, "max_features": 0.6}

# What a facilities team knows about each building type without a meter. These are coarse
# estimates, not measurements; edit them to match the college. Unknown types get flat values.
# Relative hourly activity on a term-time weekday: academic blocks peak with afternoon classes,
# libraries and hostels in the evening, labs stay high all day.
TYPE_SCHEDULES = {
    "academic": [.45] * 7 + [.5, .65, .8, .9, .9, .85, .95, 1., 1., .85, .65, .55, .5, .45, .45, .45, .45],
    "library": [.4] * 7 + [.4, .5, .6, .65, .7, .7, .75, .8, .85, .9, .95, 1., 1., .95, .85, .7, .55],
    "hostel": [.7, .65, .65, .65, .65, .7, .75, .75, .6, .5, .5, .5,
               .55, .55, .55, .6, .7, .8, .9, 1., 1., .95, .9, .8],
    "lab": [.65] * 7 + [.7, .8, .9, .95, .95, .95, .95, 1., 1., .95, .9, .85, .8, .75, .72, .7, .68],
}
# How active each type stays at weekends and during academic breaks, relative to a term weekday:
# an academic block mostly closes, hostels empty out in vacations, research labs keep running.
# The break value is only a fallback: each building's own bills give its real break level.
TYPE_CALENDAR = {  # (weekend activity, break activity)
    "academic": (0.2, 0.15),
    "library": (0.7, 0.3),
    "hostel": (1.0, 0.35),
    "lab": (0.6, 0.75),
}
FEATURES = [
    "Is_Weekend", "Is_Break", "Temperature", "Temperature_3h_Mean", "Cooling_Degrees",
    "Scheduled_Occupancy", "Occupancy_Prev_1h", "Occupancy_Next_1h",
    # The schedule scaled by the type's weekend activity and the building's break level.
    "Expected_Occupancy", "Expected_Cooling", "Weekend_Activity", "Break_Ratio",
    # Whole-day descriptors of the building type. Without them, two types whose hourly schedules
    # match at different times of day look identical (the library at 6 PM equals the lab at 2 PM),
    # and the forest copies the lab's flatter peak onto the library, leaving a dip at its peak.
    "Night_Occupancy", "Evening_Share",
]
# Clock time is deliberately not a feature: the forest would memorise when the metered buildings
# are busy and copy that onto other types instead of following each type's own schedule.


def type_schedule(building_type: str, hour: np.ndarray) -> np.ndarray:
    schedule = TYPE_SCHEDULES.get(building_type, [1.0] * 24)
    return np.interp(hour, np.arange(25), schedule + schedule[:1])


def break_ratios(bills: pd.DataFrame) -> dict[str, float]:
    """Each building's break-day energy as a fraction of a term day, read from its own bills:
    average daily kWh in all-break months over all-term months. Works for bill-only buildings
    too. A forest cannot extrapolate, so without this a building that drops more (academic) or
    less (lab) in vacations than any training building gets the nearest building's drop, and the
    load in part-break months (May, July, December) is split wrongly between term and break."""
    rows = []
    for row in bills.itertuples():
        days = pd.date_range(pd.Period(str(row.month), freq="M").start_time, periods=pd.Period(str(row.month), freq="M").days_in_month)
        rows.append({"building": row.building, "building_type": row.building_type,
                     "break_share": is_break(days).mean(), "daily_kwh": row.monthly_kWh / len(days)})
    months = pd.DataFrame(rows)
    ratios = {}
    for building, group in months.groupby("building"):
        on_break = group.loc[group["break_share"] == 1, "daily_kwh"]
        in_term = group.loc[group["break_share"] == 0, "daily_kwh"]
        fallback = TYPE_CALENDAR.get(group["building_type"].iloc[0], (1.0, 1.0))[1]
        ratios[building] = float(on_break.mean() / in_term.mean()) if len(on_break) and len(in_term) else fallback
    return ratios


def campus_temperature(interval: pd.DataFrame) -> pd.Series:
    """One campus-wide temperature reading per timestamp (all buildings share the weather)."""
    return interval.groupby("timestamp")["temperature"].mean().sort_index()


def build_features(timestamps: pd.DatetimeIndex, building_type: str, temperature: pd.Series,
                   break_ratio: float) -> pd.DataFrame:
    """Model inputs for one building over ``timestamps``. Uses only what is known without a
    meter: the weather, the academic calendar, the building type and its bills."""
    timestamps = pd.DatetimeIndex(timestamps)
    hour = np.asarray(timestamps.hour + timestamps.minute / 60, dtype=float)
    temp = temperature.reindex(timestamps).interpolate(limit_direction="both")
    temp = temp.fillna(temperature.mean()).to_numpy()
    # Buildings respond to the weather with thermal lag; a trailing mean is causal.
    temp_3h = pd.Series(temp).rolling(12, min_periods=1).mean().to_numpy()
    occupancy = type_schedule(building_type, hour)
    cooling = np.clip(temp_3h - 26.0, 0, None)
    day = type_schedule(building_type, np.arange(24, dtype=float))
    weekend_activity = TYPE_CALENDAR.get(building_type, (1.0, 1.0))[0]
    is_weekend = timestamps.day_of_week >= 5
    on_break = is_break(timestamps)  # academic calendar, known in advance
    expected = (occupancy * np.where(is_weekend, weekend_activity, 1.0)
                * np.where(on_break, break_ratio, 1.0))
    return pd.DataFrame({
        "Is_Weekend": is_weekend.astype(int),
        "Is_Break": on_break.astype(int),
        "Temperature": temp,
        "Temperature_3h_Mean": temp_3h,
        "Cooling_Degrees": cooling,
        # The timetable is known in advance, so looking an hour ahead in it is not leakage.
        "Scheduled_Occupancy": occupancy,
        "Occupancy_Prev_1h": type_schedule(building_type, (hour - 1) % 24),
        "Occupancy_Next_1h": type_schedule(building_type, (hour + 1) % 24),
        "Expected_Occupancy": expected,
        "Expected_Cooling": expected * cooling,
        "Weekend_Activity": weekend_activity,
        "Break_Ratio": break_ratio,
        # Always-on level (3 AM) and how evening-heavy the type is (5-11 PM share of the day).
        "Night_Occupancy": day[3],
        "Evening_Share": day[17:23].sum() / day.sum(),
    })


def _training_rows(interval: pd.DataFrame, bills: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if interval.attrs.get("evaluation_only"):
        raise ValueError("evaluation ground truth was passed as training data")
    temperature = campus_temperature(interval)
    observed = interval.dropna(subset=["kWh_consumed"])
    if observed.empty:
        raise ValueError("no observed 15-minute data to train on")
    ratios = break_ratios(bills)
    frames, targets = [], []
    for building, rows in observed.groupby("building"):
        rows = rows.sort_values("timestamp")
        timestamps = pd.DatetimeIndex(rows["timestamp"])
        # Average interval energy over all of the building's bills. A per-month average would hide
        # the drop in an all-vacation month (June looks "normal for June").
        own_bills = bills[bills["building"] == building]
        intervals = sum(pd.Period(str(m), freq="M").days_in_month * 96 for m in own_bills["month"])
        relative = rows["kWh_consumed"].to_numpy() / (own_bills["monthly_kWh"].sum() / intervals)
        # The break level comes from the bills, so the forest learns the load *shape* with it
        # divided out; ``shape_fn`` multiplies it back in.
        frames.append(build_features(timestamps, rows["building_type"].iloc[0], temperature, ratios[building]))
        targets.append(relative / np.where(is_break(timestamps), ratios[building], 1.0))
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def train(interval: pd.DataFrame, bills: pd.DataFrame, seed: int = SEED) -> RandomForestRegressor:
    """Fit on every real 15-minute reading in ``interval``."""
    features, target = _training_rows(interval, bills)
    model = RandomForestRegressor(**HYPERPARAMETERS, n_jobs=-1, random_state=seed)
    model.fit(features[FEATURES], target)
    return model


def shape_fn(model: RandomForestRegressor, interval: pd.DataFrame, bills: pd.DataFrame) -> ShapeFn:
    """Wrap a trained forest as a shape for ``reconstruct_building_month``."""
    temperature = campus_temperature(interval)
    ratios = break_ratios(bills)

    def predict(timestamps: pd.DatetimeIndex, building: str, building_type: str) -> np.ndarray:
        ratio = ratios.get(building, TYPE_CALENDAR.get(building_type, (1.0, 1.0))[1])
        features = build_features(timestamps, building_type, temperature, ratio)
        shape = model.predict(features[list(model.feature_names_in_)]) * np.where(is_break(timestamps), ratio, 1.0)
        return np.clip(shape, 1e-6, None)

    return predict


def metadata_path(path: Path = MODEL_PATH) -> Path:
    return Path(path).with_suffix(".meta.json")


def expected_metadata(dataset_fingerprint: str, seed: int = SEED) -> dict:
    """What a cached model must have been trained with to be reused for this dataset."""
    return {"model_version": MODEL_VERSION, "features": FEATURES, "hyperparameters": HYPERPARAMETERS,
            "seed": seed, "dataset_fingerprint": dataset_fingerprint, "calendar_breaks": BREAKS,
            "sklearn_version": sklearn.__version__}


def save(model: RandomForestRegressor, path: Path = MODEL_PATH, dataset_fingerprint: str | None = None,
         seed: int = SEED) -> Path:
    """Save the model and a metadata sidecar recording what it was trained on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = expected_metadata(dataset_fingerprint or "unknown", seed)
    provenance.write_json(metadata_path(path), {**meta, "trained_at": provenance.now_utc(),
                                                "environment": provenance.environment()})
    return path


def load(path: Path = MODEL_PATH) -> RandomForestRegressor:
    return joblib.load(path)


def load_compatible(dataset_fingerprint: str, path: Path = MODEL_PATH, seed: int = SEED) -> RandomForestRegressor | None:
    """The saved model if its metadata matches this dataset, model version, features, calendar and
    scikit-learn version; otherwise ``None`` (the caller retrains). Matching feature names alone is
    not enough: a model trained on other data has the same features."""
    path = Path(path)
    stored = provenance.read_json(metadata_path(path))
    if not path.exists() or stored is None:
        log.info("no saved model with metadata at %s", path)
        return None
    expected = expected_metadata(dataset_fingerprint, seed)
    stale = [key for key, value in expected.items() if provenance.config_hash({key: stored.get(key)}) != provenance.config_hash({key: value})]
    if stale:
        log.info("saved model at %s is stale (%s differ); retraining", path, ", ".join(stale))
        return None
    try:
        model = load(path)
    except Exception as error:  # noqa: BLE001 - any unreadable pickle (corrupt, other versions) means retrain
        log.warning("could not load saved model %s: %s", path, error)
        return None
    if list(getattr(model, "feature_names_in_", [])) != FEATURES:
        return None
    return model


if __name__ == "__main__":
    from src.data import load_data

    interval, bills = load_data()
    path = save(train(interval, bills), dataset_fingerprint=provenance.dataset_fingerprint(interval, bills))
    print(f"Trained a Random Forest on {interval['kWh_consumed'].notna().sum():,} readings; saved to {path}")
