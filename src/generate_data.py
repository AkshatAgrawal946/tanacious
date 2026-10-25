"""Create deterministic synthetic data for the Missing Meter demo."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BUILDINGS = {
    "B1_Hostel_A": ("hostel", 40, 60),
    "B2_Hostel_B": ("hostel", 35, 55),
    "B3_Academic": ("academic", 30, 25),
    "B4_Library": ("library", 25, 30),
    "B5_Lab": ("lab", 50, 20),
}
INTERVAL_METERED = {"academic", "lab"}


def _temperature(ts: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hour = ts.hour + ts.minute / 60
    return 30 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5, len(ts))


def _load(ts: pd.DatetimeIndex, temp: np.ndarray, base: float, peak: float, rng: np.random.Generator) -> np.ndarray:
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    evening = peak * np.exp(-((hour - 18) ** 2) / 4)
    daytime = 0.4 * peak * ((hour >= 9) & (hour < 18))
    cooling = 1.5 * np.clip(temp - 26, 0, None)
    weekend = np.where(ts.dayofweek >= 5, 0.8, 1.0)
    return np.clip((base + evening + daytime + cooling) * weekend + rng.normal(0, 3, len(ts)), 0, None) / 4


def _dropouts(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = values.copy()
    for _ in range(3):
        size = int(rng.integers(4, 49))  # one to twelve hours
        start = int(rng.integers(0, len(result) - size))
        result[start:start + size] = np.nan
    return result


def generate(output_dir: str | Path = "data/raw", seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return interval observations, bills, and hidden evaluation truth."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-01", "2026-08-31 23:45", freq="15min")
    temp = _temperature(ts, rng)
    full = pd.concat([
        pd.DataFrame({
            "timestamp": ts, "building": name, "building_type": kind,
            "temperature": temp.round(1), "kWh_consumed": _load(ts, temp, base, peak, rng).round(3),
        })
        for name, (kind, base, peak) in BUILDINGS.items()
    ], ignore_index=True)
    bills = (full.assign(month=full["timestamp"].dt.to_period("M").astype(str))
             .groupby(["building", "building_type", "month"], as_index=False)["kWh_consumed"].sum()
             .rename(columns={"kWh_consumed": "monthly_kWh"}))
    interval = full[full["building_type"].isin(INTERVAL_METERED)].copy()
    interval["kWh_consumed"] = interval.groupby("building")["kWh_consumed"].transform(lambda s: _dropouts(s.to_numpy(), rng))
    hidden = full[~full["building_type"].isin(INTERVAL_METERED)].copy()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    interval.to_csv(output / "interval_15min.csv", index=False)
    bills.to_csv(output / "monthly_totals.csv", index=False)
    hidden.to_csv(output / "ground_truth_15min.csv", index=False)
    return interval, bills, hidden


if __name__ == "__main__":
    interval, bills, _ = generate()
    print(f"Generated {len(interval):,} interval observations and {len(bills):,} monthly bills.")
