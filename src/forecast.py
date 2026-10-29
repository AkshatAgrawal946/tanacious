"""48-hour campus load forecast (FORECAST: turns the dashboard from an audit tool into a planning tool).

The Random Forest already knows how each building type responds to the weather, the timetable
and the academic calendar. To look ahead we feed it a weather forecast instead of recorded
weather. There is no live weather feed in this MVP, so ``simulate_weather`` stands in for one:
it takes the campus's own temperature climatology for the forecast dates and adds a seeded
day-to-day anomaly. Swap it for a real forecast API without touching the rest.

Information availability: ``ml_model`` warns that ``break_ratios`` reads every bill, including
months after the one being modelled. Here only bills for months that finished before the forecast
starts are used (``completed_bills``), so the forecast never sees a bill from its own future.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.campus_calendar import is_break
from src.ml_model import TYPE_CALENDAR, break_ratios, build_features, campus_temperature

HORIZON_HOURS = 48
WINDOW_DAYS = 7  # climatology: same time of day within +/- this many days of the year


def completed_bills(bills: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Bills for months that ended before ``start``. Falls back to all bills when none have
    finished yet (e.g. a forecast for a date before the dataset begins)."""
    done = bills[pd.PeriodIndex(bills["month"].astype(str), freq="M").end_time < start]
    return done if not done.empty else bills


def simulate_weather(interval: pd.DataFrame, timestamps: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    """Mock weather forecast: the campus's average temperature at each time of day around these
    dates of the year, plus one random warm/cool anomaly per day and a little 15-minute noise."""
    history = campus_temperature(interval)
    hist_doy = history.index.day_of_year.to_numpy()
    hist_slot = (history.index.hour * 4 + history.index.minute // 15).to_numpy()
    rng = np.random.default_rng(seed)
    anomaly = {day: rng.normal(0, 1.5) for day in timestamps.normalize().unique()}
    values = []
    for ts in timestamps:
        # Circular distance in days so a late-December forecast still draws on early January.
        gap = np.abs(hist_doy - ts.day_of_year)
        near = (np.minimum(gap, 365 - gap) <= WINDOW_DAYS) & (hist_slot == ts.hour * 4 + ts.minute // 15)
        values.append(history.to_numpy()[near].mean() + anomaly[ts.normalize()])
    return pd.Series(np.asarray(values) + rng.normal(0, 0.3, len(timestamps)), index=timestamps, name="temperature")


def generate_forecast(interval: pd.DataFrame, bills: pd.DataFrame, model: RandomForestRegressor,
                      start: pd.Timestamp, hours: int = HORIZON_HOURS,
                      quantiles: tuple[float, float] = (0.1, 0.9)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Projected 15-minute load for every building over the next ``hours`` from ``start``.

    Returns ``(by_building, campus)``: per-building kW, and the campus total with a low/high band
    from the spread of the forest's individual trees (model uncertainty only, not weather error).
    """
    start = pd.Timestamp(start)
    timestamps = pd.date_range(start, periods=hours * 4, freq="15min")
    weather = simulate_weather(interval, timestamps, seed=int(start.strftime("%Y%m%d")))
    known = completed_bills(bills, start)
    ratios = break_ratios(known)
    on_break = np.where(is_break(timestamps), 1.0, 0.0)

    frames, tree_totals = [], np.zeros((len(model.estimators_), len(timestamps)))
    for building, own in known.groupby("building"):
        building_type = own["building_type"].iloc[0]
        ratio = ratios.get(building, TYPE_CALENDAR.get(building_type, (1.0, 1.0))[1])
        features = build_features(timestamps, building_type, weather, ratio)[list(model.feature_names_in_)]
        # The forest predicts load relative to the building's average 15-minute energy (see
        # ml_model); the completed bills give that average, which turns the shape back into kW.
        intervals = sum(pd.Period(str(m), freq="M").days_in_month * 96 for m in own["month"])
        to_kw = own["monthly_kWh"].sum() / intervals * 4 * np.where(on_break, ratio, 1.0)
        per_tree = np.stack([tree.predict(features.to_numpy()) for tree in model.estimators_]) * to_kw
        tree_totals += per_tree
        frames.append(pd.DataFrame({"timestamp": timestamps, "building": building, "building_type": building_type,
                                    "kW": np.clip(per_tree.mean(axis=0), 0, None)}))

    by_building = pd.concat(frames, ignore_index=True)
    low, high = np.quantile(tree_totals, quantiles, axis=0)
    campus = pd.DataFrame({"timestamp": timestamps, "kW": by_building.groupby("timestamp")["kW"].sum().to_numpy(),
                           "kW_low": np.clip(low, 0, None), "kW_high": high, "temperature": weather.to_numpy()})
    return by_building, campus
