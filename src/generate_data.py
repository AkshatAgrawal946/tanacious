"""Generate a synthetic electricity dataset for 5 campus buildings over 3 months.

Metering resolution differs by building type:
  - academic blocks and labs have smart meters  -> 15-minute readings
  - hostels and libraries only have a utility bill -> monthly totals

Outputs:
  data/raw/interval_15min.csv     - 15-min readings for academic/lab buildings, with short
                                    sensor-dropout gaps (NaN kWh)
  data/raw/monthly_totals.csv     - monthly kWh for every building (the bill)
  data/raw/ground_truth_15min.csv - hidden 15-min series for the monthly-only buildings;
                                    for evaluating disaggregation only, never as model input
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2026-06-01", "2026-08-31 23:45"
FREQ = "15min"
STEPS_PER_HOUR = 4
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (building_type, base load kWh/h, evening peak extra kWh/h)
BUILDINGS = {
    "B1_Hostel_A": ("hostel", 40, 60),
    "B2_Hostel_B": ("hostel", 35, 55),
    "B3_Academic": ("academic", 30, 25),
    "B4_Library": ("library", 25, 30),
    "B5_Lab": ("lab", 50, 20),
}
INTERVAL_METERED = {"academic", "lab"}  # everything else only has monthly totals


def make_temperature(ts: pd.DatetimeIndex, rng) -> np.ndarray:
    """Summer temperature: daily cycle peaking ~3 PM, plus noise."""
    hour = ts.hour + ts.minute / 60
    daily = 6 * np.sin(2 * np.pi * (hour - 9) / 24)
    return 30 + daily + rng.normal(0, 1.5, len(ts))


def make_load(ts, temp, base, peak, rng) -> np.ndarray:
    """kWh consumed in each 15-min interval."""
    hour = (ts.hour + ts.minute / 60).to_numpy()
    # 4 PM - 8 PM spike (smooth bump centred at 6 PM)
    evening = peak * np.exp(-((hour - 18) ** 2) / 4)
    daytime = 0.4 * peak * ((hour >= 9) & (hour < 18))
    cooling = 1.5 * np.clip(temp - 26, 0, None)  # AC load above 26 C
    weekend = np.where(ts.dayofweek >= 5, 0.8, 1.0)
    load_kw = (base + evening + daytime + cooling) * weekend + rng.normal(0, 3, len(ts))
    return np.clip(load_kw, 0, None) / STEPS_PER_HOUR


def add_missing_chunks(kwh: np.ndarray, rng, n_chunks=3, min_hours=1, max_hours=12) -> np.ndarray:
    """Simulate smart-meter dropouts: blank out a few short runs of readings."""
    kwh = kwh.copy()
    for _ in range(n_chunks):
        length = rng.integers(min_hours, max_hours + 1) * STEPS_PER_HOUR
        start = rng.integers(0, len(kwh) - length)
        kwh[start : start + length] = np.nan
    return kwh


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, END, freq=FREQ)
    temp = make_temperature(ts, rng)

    frames = []
    for name, (btype, base, peak) in BUILDINGS.items():
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "building": name,
            "building_type": btype,
            "temperature": temp.round(1),
            "kWh_consumed": make_load(ts, temp, base, peak, rng).round(3),
        }))
    full = pd.concat(frames, ignore_index=True)

    # Monthly totals come from the complete data (the bill is always known)
    monthly = (full.groupby(["building", "building_type", full["timestamp"].dt.to_period("M")])["kWh_consumed"]
               .sum().round(1).reset_index().rename(columns={"timestamp": "month", "kWh_consumed": "monthly_kWh"}))

    metered = full["building_type"].isin(INTERVAL_METERED)
    interval = full[metered].reset_index(drop=True)
    hidden = full[~metered].reset_index(drop=True)

    interval["kWh_consumed"] = interval.groupby("building")["kWh_consumed"].transform(
        lambda s: add_missing_chunks(s.to_numpy(), rng))
    return interval, monthly, hidden


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    interval, monthly, hidden = generate()
    interval.to_csv(OUT_DIR / "interval_15min.csv", index=False)
    monthly.to_csv(OUT_DIR / "monthly_totals.csv", index=False)
    hidden.to_csv(OUT_DIR / "ground_truth_15min.csv", index=False)
    print(f"15-min: {len(interval):,} rows ({interval['building'].nunique()} buildings), "
          f"{interval['kWh_consumed'].isna().mean():.2%} missing kWh")
    print(f"monthly: {len(monthly)} rows ({monthly['building'].nunique()} buildings)")
    print(f"ground truth (eval only): {len(hidden):,} rows -> {OUT_DIR}")
