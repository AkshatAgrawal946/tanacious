import numpy as np
import pandas as pd

from src import ml_model
from src.compare import compare_models
from src.generate_data import PILOT_METERS, generate
from src.reconstruct import reconcile_monthly, reconstruct_building_month


def test_reconcile_monthly_is_exact():
    output = reconcile_monthly(np.array([1.0, 2.0, 3.0]), 55.0)
    assert np.isclose(output.sum(), 55.0)
    assert (output >= 0).all()


def test_reconstruction_matches_bill_and_has_15_minute_intervals(tmp_path):
    interval, bills, _ = generate(tmp_path)
    result = reconstruct_building_month(interval, bills, "B2_Hostel_B", "2026-06")  # no pilot meter
    bill = bills[(bills.building == "B2_Hostel_B") & (bills.month == "2026-06")].iloc[0]
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
    metered = interval[~interval.building.isin(hidden.building)]  # pilot weeks are already in hidden
    full = pd.concat([metered, hidden]).dropna(subset=["kWh_consumed"])
    peaks = {name: _weekday_peak_hour(group) for name, group in full.groupby("building")}
    assert 14 <= peaks["B3_Academic"] < 16  # afternoon classes
    assert 17 <= peaks["B4_Library"] < 20  # evening study
    assert 18 <= peaks["B1_Hostel_A"] < 21 and 18 <= peaks["B2_Hostel_B"] < 21
    campus = full.groupby("timestamp", as_index=False)["kWh_consumed"].sum()
    assert 16 <= _weekday_peak_hour(campus) < 20  # campus spike stays in the 4-8 PM window


def test_pilot_meters_release_only_their_window_and_are_not_scored(tmp_path):
    interval, bills, hidden = generate(tmp_path)
    for building, (start, end) in PILOT_METERS.items():
        pilot = interval[interval.building == building]
        assert len(pilot) == 14 * 96
        assert pilot.timestamp.min() == pd.Timestamp(start)
        assert pilot.timestamp.max() == pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=45)
    series, metrics = compare_models(interval, bills, hidden)
    hostel = series[series.building == "B1_Hostel_A"]
    assert hostel.is_pilot.sum() == 14 * 96
    assert metrics.loc[metrics.building == "B1_Hostel_A", "evaluated_against"].eq("hidden ground truth").all()


def test_random_forest_reconstruction_matches_bill(tmp_path):
    interval, bills, _ = generate(tmp_path)
    shape = ml_model.shape_fn(ml_model.train(interval, bills), interval)
    for building in ("B1_Hostel_A", "B3_Academic"):
        result = reconstruct_building_month(interval, bills, building, "2026-06", shape)
        bill = bills[(bills.building == building) & (bills.month == "2026-06")].iloc[0]
        assert np.isclose(result.kWh_consumed.sum(), bill.monthly_kWh)
        assert (result.kWh_consumed >= 0).all()


def test_model_comparison_scores_every_building_with_both_models(tmp_path):
    interval, bills, hidden = generate(tmp_path)
    series, metrics = compare_models(interval, bills, hidden)
    assert set(metrics.building) == set(bills.building)
    assert set(metrics.model) == {"Baseline", "Random forest"}
    assert series[["Baseline", "Random forest"]].notna().all().all()
    forest = metrics[metrics.model == "Random forest"].set_index("building")
    assert 17 <= forest.loc["B1_Hostel_A", "predicted_peak_hour"] < 21


def test_unmetered_types_are_retimed_to_their_own_peak(tmp_path):
    interval, bills, _ = generate(tmp_path)
    for building in ("B1_Hostel_A", "B4_Library"):
        result = reconstruct_building_month(interval, bills, building, "2026-06")
        assert 17 <= _weekday_peak_hour(result) < 21
