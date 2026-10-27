"""Create deterministic synthetic data for the Missing Meter demo: one calendar year (2026)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.campus_calendar import is_break

SEED = 42
START, END = "2026-01-01", "2026-12-31 23:45"
BUILDINGS = {
    "B1_Hostel_A": ("hostel", 40, 60),
    "B2_Hostel_B": ("hostel", 35, 55),
    "B3_Academic": ("academic", 30, 25),
    "B4_Library": ("library", 25, 30),
    "B5_Lab": ("lab", 50, 20),
}
INTERVAL_METERED = {"academic", "lab"}
# A temporary clamp-on meter on bill-only buildings: their 15-minute readings are released for
# this window only (inclusive); the rest of the year stays bill-only. Placed in term time so the
# readings show normal use rather than the summer vacation.
PILOT_METERS = {"B1_Hostel_A": ("2026-08-03", "2026-08-16"), "B4_Library": ("2026-08-03", "2026-08-16")}

# Weekday occupancy (0-1) for each hour, by building type. Each type peaks at its own time:
# academic during afternoon classes (2-4 PM), library and hostels in the evening, labs flat.
OCCUPANCY = {
    "academic": [.05, .05, .05, .05, .05, .05, .05, .1, .35, .6, .75, .8,
                 .6, .8, .95, 1., .8, .45, .2, .1, .05, .05, .05, .05],
    "library": [.1, .02, .02, .02, .02, .02, .02, .05, .2, .35, .4, .45,
                .45, .5, .55, .6, .7, .85, .95, 1., .9, .7, .45, .25],
    "hostel": [.65, .55, .5, .5, .5, .55, .7, .8, .55, .3, .2, .2,
               .3, .3, .2, .25, .4, .6, .85, 1., 1., .95, .9, .8],
    "lab": [.35, .35, .35, .35, .35, .35, .35, .45, .6, .8, .9, .95,
            .85, .9, .95, 1., .95, .85, .7, .6, .5, .45, .4, .38],
}
# Weekend occupancy as a fraction of the weekday schedule.
WEEKEND_FACTOR = {"academic": 0.15, "library": 0.7, "hostel": 1.0, "lab": 0.6}
# Occupancy during academic breaks as a fraction of term time: classes stop, most students go
# home, research labs keep running.
BREAK_FACTOR = {"academic": 0.1, "library": 0.25, "hostel": 0.3, "lab": 0.7}


def _temperature(ts: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hour = ts.hour + ts.minute / 60
    days = ts.normalize()
    day_idx = days.unique().get_indexer(days)
    # Seasonal daily mean: hottest in late May (~33 C), coolest in winter (~21 C), plus day-to-day weather.
    season = 27 + 6 * np.cos(2 * np.pi * (ts.day_of_year - 140) / 365)
    weather = rng.normal(0, 1.2, day_idx.max() + 1)[day_idx]
    return season + weather + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5, len(ts))


def _occupancy(ts: pd.DatetimeIndex, kind: str) -> np.ndarray:
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    schedule = OCCUPANCY[kind]
    weekday = np.interp(hour, np.arange(25), schedule + schedule[:1])
    calendar = np.where(is_break(ts), BREAK_FACTOR[kind], 1.0)
    return np.where(ts.day_of_week >= 5, WEEKEND_FACTOR[kind], 1.0) * calendar * weekday


def _load(ts: pd.DatetimeIndex, temp: np.ndarray, kind: str, base: float, peak: float, rng: np.random.Generator) -> np.ndarray:
    occupancy = _occupancy(ts, kind)
    # Cooling is mostly switched on for occupied space, so an empty building barely cools.
    cooling = 1.5 * np.clip(temp - 26, 0, None) * (0.3 + 0.7 * occupancy)
    return np.clip(base + peak * occupancy + cooling + rng.normal(0, 3, len(ts)), 0, None) / 4


def _dropouts(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = values.copy()
    for _ in range(12):  # about one sensor dropout a month
        size = int(rng.integers(4, 49))  # one to twelve hours
        start = int(rng.integers(0, len(result) - size))
        result[start:start + size] = np.nan
    return result


def generate(output_dir: str | Path = "data/raw", seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return interval observations, bills, and hidden evaluation truth."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, END, freq="15min")
    temp = _temperature(ts, rng)
    full = pd.concat([
        pd.DataFrame({
            "timestamp": ts, "building": name, "building_type": kind,
            "temperature": temp.round(1), "kWh_consumed": _load(ts, temp, kind, base, peak, rng).round(3),
        })
        for name, (kind, base, peak) in BUILDINGS.items()
    ], ignore_index=True)
    bills = (full.assign(month=full["timestamp"].dt.to_period("M").astype(str))
             .groupby(["building", "building_type", "month"], as_index=False)["kWh_consumed"].sum()
             .rename(columns={"kWh_consumed": "monthly_kWh"}))
    interval = full[full["building_type"].isin(INTERVAL_METERED)].copy()
    interval["kWh_consumed"] = interval.groupby("building")["kWh_consumed"].transform(lambda s: _dropouts(s.to_numpy(), rng))
    hidden = full[~full["building_type"].isin(INTERVAL_METERED)].copy()
    pilots = [hidden[(hidden["building"] == building) & (hidden["timestamp"] >= start)
                     & (hidden["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
              for building, (start, end) in PILOT_METERS.items()]
    interval = pd.concat([interval, *pilots], ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    interval.to_csv(output / "interval_15min.csv", index=False)
    bills.to_csv(output / "monthly_totals.csv", index=False)
    hidden.to_csv(output / "ground_truth_15min.csv", index=False)
    return interval, bills, hidden


if __name__ == "__main__":
    interval, bills, _ = generate()
    print(f"Generated {len(interval):,} interval observations and {len(bills):,} monthly bills.")
