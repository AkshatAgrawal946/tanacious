import numpy as np
import pandas as pd

from src.generate_data import generate
from src.reconstruct import reconcile_monthly, reconstruct_building_month


def test_reconcile_monthly_is_exact():
    output = reconcile_monthly(np.array([1.0, 2.0, 3.0]), 55.0)
    assert np.isclose(output.sum(), 55.0)
    assert (output >= 0).all()


def test_reconstruction_matches_bill_and_has_15_minute_intervals(tmp_path):
    interval, bills, _ = generate(tmp_path)
    result = reconstruct_building_month(interval, bills, "B1_Hostel_A", "2026-06")
    bill = bills[(bills.building == "B1_Hostel_A") & (bills.month == "2026-06")].iloc[0]
    assert len(result) == 30 * 96
    assert result["kWh_consumed"].notna().all()
    assert result["is_reconstructed"].all()
    assert np.isclose(result["kWh_consumed"].sum(), bill.monthly_kWh)


def test_reconstruction_preserves_observed_readings(tmp_path):
    interval, bills, _ = generate(tmp_path)
    result = reconstruct_building_month(interval, bills, "B3_Academic", "2026-06")
    observed = interval[(interval.building == "B3_Academic") & interval.kWh_consumed.notna()]
    observed = observed[observed.timestamp.dt.to_period("M").astype(str) == "2026-06"]
    first = observed.iloc[0]
    actual = result.loc[result.timestamp == first.timestamp].iloc[0]
    bill = bills[(bills.building == "B3_Academic") & (bills.month == "2026-06")].iloc[0]
    assert actual.is_observed
    assert not actual.is_reconstructed
    assert actual.kWh_consumed == first.kWh_consumed
    assert np.isclose(result.kWh_consumed.sum(), bill.monthly_kWh)


def _weekday_peak_hour(frame: pd.DataFrame) -> int:
    weekday = frame[frame.timestamp.dt.day_of_week < 5]
    return int(weekday.groupby(weekday.timestamp.dt.hour)["kWh_consumed"].mean().idxmax())


def test_each_building_type_peaks_at_its_own_time(tmp_path):
    interval, _, hidden = generate(tmp_path)
    full = pd.concat([interval, hidden]).dropna(subset=["kWh_consumed"])
    peaks = {name: _weekday_peak_hour(group) for name, group in full.groupby("building")}
    assert 14 <= peaks["B3_Academic"] < 16  # afternoon classes
    assert 17 <= peaks["B4_Library"] < 20  # evening study
    assert 18 <= peaks["B1_Hostel_A"] < 21 and 18 <= peaks["B2_Hostel_B"] < 21
    campus = full.groupby("timestamp", as_index=False)["kWh_consumed"].sum()
    assert 16 <= _weekday_peak_hour(campus) < 20  # campus spike stays in the 4-8 PM window


def test_unmetered_types_are_retimed_to_their_own_peak(tmp_path):
    interval, bills, _ = generate(tmp_path)
    for building in ("B1_Hostel_A", "B4_Library"):
        result = reconstruct_building_month(interval, bills, building, "2026-06")
        assert 17 <= _weekday_peak_hour(result) < 21
