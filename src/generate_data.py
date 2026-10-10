"""Generate a synthetic hourly electricity dataset for 5 campus buildings over 3 months.

Outputs:
  data/raw/hourly_consumption.csv  - hourly readings, with random multi-day gaps (NaN kWh)
  data/raw/monthly_totals.csv      - true monthly kWh per building (what the meter bill shows)

The gaps simulate buildings where only the monthly total is reliably known.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2026-06-01", "2026-08-31 23:00"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (building_type, base load kWh, evening peak extra kWh)
BUILDINGS = {
    "B1_Hostel_A": ("hostel", 40, 60),
    "B2_Hostel_B": ("hostel", 35, 55),
    "B3_Academic": ("academic", 30, 25),
    "B4_Library": ("library", 25, 30),
    "B5_Lab": ("lab", 50, 20),
}


def make_temperature(ts: pd.DatetimeIndex, rng) -> np.ndarray:
    """Summer temperature: daily cycle peaking ~3 PM, plus noise."""
    daily = 6 * np.sin(2 * np.pi * (ts.hour - 9) / 24)
    return 30 + daily + rng.normal(0, 1.5, len(ts))


def make_load(ts, temp, base, peak, rng) -> np.ndarray:
    hour = ts.hour.to_numpy()
    # 4 PM - 8 PM spike (smooth bump centred at 6 PM)
    evening = peak * np.exp(-((hour - 18) ** 2) / 4)
    daytime = 0.4 * peak * ((hour >= 9) & (hour <= 17))
    cooling = 1.5 * np.clip(temp - 26, 0, None)  # AC load above 26 C
    weekend = np.where(ts.dayofweek >= 5, 0.8, 1.0)
    load = (base + evening + daytime + cooling) * weekend
    return np.clip(load + rng.normal(0, 3, len(ts)), 0, None)


def add_missing_chunks(kwh: np.ndarray, rng, n_chunks=3, min_days=2, max_days=7) -> np.ndarray:
    kwh = kwh.copy()
    for _ in range(n_chunks):
        length = rng.integers(min_days, max_days + 1) * 24
        start = rng.integers(0, len(kwh) - length)
        kwh[start : start + length] = np.nan
    return kwh


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, END, freq="h")
    temp = make_temperature(ts, rng)

    frames = []
    for name, (btype, base, peak) in BUILDINGS.items():
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "building": name,
            "building_type": btype,
            "temperature": temp.round(1),
            "kWh_consumed": make_load(ts, temp, base, peak, rng).round(2),
        }))
    full = pd.concat(frames, ignore_index=True)

    # Monthly totals come from the complete data (the "bill" is always known)
    monthly = (full.groupby(["building", full["timestamp"].dt.to_period("M")])["kWh_consumed"]
               .sum().round(1).reset_index().rename(columns={"timestamp": "month", "kWh_consumed": "monthly_kWh"}))

    # Knock out random multi-day chunks of hourly data per building
    full["kWh_consumed"] = full.groupby("building")["kWh_consumed"].transform(
        lambda s: add_missing_chunks(s.to_numpy(), rng))
    return full, monthly


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly, monthly = generate()
    hourly.to_csv(OUT_DIR / "hourly_consumption.csv", index=False)
    monthly.to_csv(OUT_DIR / "monthly_totals.csv", index=False)
    print(f"hourly: {len(hourly):,} rows, {hourly['kWh_consumed'].isna().mean():.1%} missing kWh")
    print(f"monthly: {len(monthly)} rows -> {OUT_DIR}")
