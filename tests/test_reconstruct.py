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
