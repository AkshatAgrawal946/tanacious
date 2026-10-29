"""Accounting invariants of bill-constrained reconstruction across month lengths and metering levels."""
import numpy as np
import pandas as pd
import pytest

from src.data import BILL_TOLERANCE_KWH
from src.reconstruct import reconcile_monthly, reconstruct_building_month


def _shape(timestamps, building, building_type):
    hour = timestamps.hour + timestamps.minute / 60
    return 1 + np.sin(2 * np.pi * hour / 24) ** 2


def _case(month: str, observed_fraction: float, bill: float, seed: int = 0):
    period = pd.Period(month, freq="M")
    ts = pd.date_range(period.start_time, period.end_time, freq="15min")
    rng = np.random.default_rng(seed)
    observed = np.sort(rng.choice(len(ts), size=int(observed_fraction * len(ts)), replace=False))
    readings = rng.uniform(0.0, 2.0, len(observed))
    if observed_fraction == 1.0:
        readings *= bill / readings.sum() if readings.sum() else 0
    interval = pd.DataFrame({"timestamp": ts[observed], "building": "B", "building_type": "lab", "temperature": 20.0,
                             "kWh_consumed": readings})
    bills = pd.DataFrame([{"building": "B", "building_type": "lab", "month": month, "monthly_kWh": bill}])
    return interval, bills, ts


@pytest.mark.parametrize("month, days", [("2026-02", 28), ("2028-02", 29), ("2026-04", 30), ("2026-01", 31)])
@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.5, 0.95, 1.0])
def test_reconstruction_invariants(month, days, fraction):
    interval, bills, ts = _case(month, fraction, bill=days * 96 * 1.5)
    result = reconstruct_building_month(interval, bills, "B", month, _shape)
    assert len(result) == days * 96 and result["timestamp"].equals(pd.Series(ts, name="timestamp"))
    values = result["kWh_consumed"].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    assert abs(values.sum() - bills.monthly_kWh.iloc[0]) <= BILL_TOLERANCE_KWH
    known = interval.set_index("timestamp")["kWh_consumed"]
    observed = result[result["is_observed"]].set_index("timestamp")["kWh_consumed"]
    assert observed.index.equals(known.sort_index().index)
    assert (observed == known.reindex(observed.index)).all()  # bit-for-bit unchanged
    assert (result["is_observed"] ^ result["is_reconstructed"]).all()


def test_zero_consumption_month():
    interval, bills, _ = _case("2026-06", 0.3, bill=0.0)
    interval["kWh_consumed"] = 0.0
    result = reconstruct_building_month(interval, bills, "B", "2026-06", _shape)
    assert (result["kWh_consumed"] == 0).all()


def test_observed_slightly_above_bill_within_tolerance():
    interval, bills, _ = _case("2026-06", 0.5, bill=1.0)
    bills["monthly_kWh"] = interval["kWh_consumed"].sum() - BILL_TOLERANCE_KWH / 2
    result = reconstruct_building_month(interval, bills, "B", "2026-06", _shape)
    assert (result.loc[result.is_reconstructed, "kWh_consumed"] == 0).all()


def test_observed_above_bill_fails():
    interval, bills, _ = _case("2026-06", 0.5, bill=1.0)
    with pytest.raises(ValueError, match="exceeds the monthly bill"):
        reconstruct_building_month(interval, bills, "B", "2026-06", _shape)


def test_fully_observed_month_that_disagrees_with_bill_fails():
    interval, bills, _ = _case("2026-06", 1.0, bill=100.0)
    with pytest.raises(ValueError, match="do not reconcile"):
        reconstruct_building_month(interval, bills.assign(monthly_kWh=200.0), "B", "2026-06", _shape)


def test_duplicate_observations_fail():
    interval, bills, _ = _case("2026-06", 0.2, bill=5000.0)
    with pytest.raises(ValueError, match="duplicate meter readings"):
        reconstruct_building_month(pd.concat([interval, interval.iloc[[0]]]), bills, "B", "2026-06", _shape)


def test_duplicate_bills_fail():
    interval, bills, _ = _case("2026-06", 0.2, bill=5000.0)
    with pytest.raises(ValueError, match="expected one bill"):
        reconstruct_building_month(interval, pd.concat([bills, bills]), "B", "2026-06", _shape)


def test_off_grid_readings_fail():
    interval, bills, _ = _case("2026-06", 0.2, bill=5000.0)
    interval.loc[interval.index[0], "timestamp"] += pd.Timedelta("7min")
    with pytest.raises(ValueError, match="off the 15-minute grid"):
        reconstruct_building_month(interval, bills, "B", "2026-06", _shape)


@pytest.mark.parametrize("bad_shape, message", [
    (lambda ts, b, t: np.ones(len(ts) - 1), "returned"),
    (lambda ts, b, t: np.full(len(ts), np.nan), "non-finite"),
])
def test_bad_shape_fails(bad_shape, message):
    interval, bills, _ = _case("2026-06", 0.2, bill=5000.0)
    with pytest.raises(ValueError, match=message):
        reconstruct_building_month(interval, bills, "B", "2026-06", bad_shape)


@pytest.mark.parametrize("total", [0.0, 1e-9, 1.0, 12345.678, 1e7])
def test_reconcile_monthly_never_negative_and_exact(total):
    shape = np.random.default_rng(1).uniform(1e-6, 1, 2976)
    out = reconcile_monthly(shape, total)
    assert (out >= 0).all() and np.isclose(out.sum(), total, rtol=0, atol=1e-9 * max(total, 1))
