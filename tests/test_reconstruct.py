import numpy as np
import pandas as pd
import pytest

from src import ml_model
from src.campus_calendar import is_break
from src.evaluate import evaluate_model
from src.generate_data import PILOT_METERS, generate
from src.reconstruct import reconcile_monthly, reconstruct_building_month


@pytest.fixture(scope="module")
def campus(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def forest(campus):
    interval, bills, _ = campus
    return ml_model.shape_fn(ml_model.train(interval, bills), interval, bills)


@pytest.fixture(scope="module")
def evaluation(campus):
    interval, bills, hidden = campus
    return evaluate_model(interval, bills, hidden)


def _weekday_peak_hour(frame: pd.DataFrame, column: str = "kWh_consumed") -> int:
    weekday = frame[frame.timestamp.dt.day_of_week < 5]
    return int(weekday.groupby(weekday.timestamp.dt.hour)[column].mean().idxmax())


def _full_truth(interval: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    metered = interval[~interval.building.isin(hidden.building)]  # pilot weeks are already in hidden
    return pd.concat([metered, hidden]).dropna(subset=["kWh_consumed"])


def test_reconcile_monthly_is_exact():
    output = reconcile_monthly(np.array([1.0, 2.0, 3.0]), 55.0)
    assert np.isclose(output.sum(), 55.0)
    assert (output >= 0).all()


def test_reconstruction_matches_bill_and_has_15_minute_intervals(campus, forest):
    interval, bills, _ = campus
    result = reconstruct_building_month(interval, bills, "B2_Hostel_B", "2026-06", forest)  # no pilot meter
    bill = bills[(bills.building == "B2_Hostel_B") & (bills.month == "2026-06")].iloc[0]
    assert len(result) == 30 * 96
    assert result["kWh_consumed"].notna().all()
    assert result["is_reconstructed"].all()
    assert (result["kWh_consumed"] >= 0).all()
    assert np.isclose(result["kWh_consumed"].sum(), bill.monthly_kWh)


def test_reconstruction_preserves_observed_readings(campus, forest):
    interval, bills, _ = campus
    result = reconstruct_building_month(interval, bills, "B3_Academic", "2026-06", forest)
    observed = interval[(interval.building == "B3_Academic") & interval.kWh_consumed.notna()]
    observed = observed[observed.timestamp.dt.to_period("M").astype(str) == "2026-06"]
    first = observed.iloc[0]
    actual = result.loc[result.timestamp == first.timestamp].iloc[0]
    bill = bills[(bills.building == "B3_Academic") & (bills.month == "2026-06")].iloc[0]
    assert actual.is_observed
    assert not actual.is_reconstructed
    assert actual.kWh_consumed == first.kWh_consumed
    assert np.isclose(result.kWh_consumed.sum(), bill.monthly_kWh)


def test_each_building_type_peaks_at_its_own_time(campus):
    interval, _, hidden = campus
    full = _full_truth(interval, hidden)
    peaks = {name: _weekday_peak_hour(group) for name, group in full.groupby("building")}
    assert 14 <= peaks["B3_Academic"] < 16  # afternoon classes
    assert 17 <= peaks["B4_Library"] < 20  # evening study
    assert 18 <= peaks["B1_Hostel_A"] < 21 and 18 <= peaks["B2_Hostel_B"] < 21
    campus_total = full.groupby("timestamp", as_index=False)["kWh_consumed"].sum()
    assert 16 <= _weekday_peak_hour(campus_total) < 20  # campus spike stays in the 4-8 PM window


def test_pilot_meters_release_only_their_window_and_are_not_scored(campus, evaluation):
    interval, _, _ = campus
    for building, windows in PILOT_METERS.items():
        pilot = interval[interval.building == building]
        assert len(pilot) == len(windows) * 14 * 96
        for start, end in windows:
            window = pilot[(pilot.timestamp >= start) & (pilot.timestamp < pd.Timestamp(end) + pd.Timedelta(days=1))]
            assert len(window) == 14 * 96
    # One pilot fortnight falls in a break, so the model sees a near-empty building too.
    assert is_break(interval[interval.building == "B1_Hostel_A"].timestamp).any()
    series, metrics = evaluation
    assert series[series.building == "B1_Hostel_A"].is_pilot.sum() == 2 * 14 * 96
    assert metrics.set_index("building").loc["B1_Hostel_A", "evaluated_against"] == "hidden ground truth"


def test_evaluation_scores_every_building(campus, evaluation):
    _, bills, _ = campus
    series, metrics = evaluation
    assert set(metrics.building) == set(bills.building)
    assert series["predicted"].notna().all()
    scores = metrics.set_index("building")
    assert 17 <= scores.loc["B1_Hostel_A", "predicted_peak_hour"] < 21
    assert scores.loc[["B1_Hostel_A", "B2_Hostel_B", "B4_Library"], "mape_pct"].lt(12).all()


def test_unmetered_types_get_their_own_evening_peak(campus, forest):
    interval, bills, _ = campus
    for building in ("B1_Hostel_A", "B4_Library"):
        result = reconstruct_building_month(interval, bills, building, "2026-09", forest)  # term time
        assert 17 <= _weekday_peak_hour(result) < 21


def test_random_forest_has_no_dip_at_the_library_peak(evaluation):
    series, _ = evaluation
    library = series[(series.building == "B4_Library") & ~series.is_pilot & (series.timestamp.dt.day_of_week < 5)]
    hour = library.timestamp.dt.hour + library.timestamp.dt.minute / 60
    profile = library.groupby(hour)["predicted"].mean()
    # Library schedules at 6-7 PM match the lab's at 2-3 PM; the forest must not copy the lab's flatter peak.
    assert profile.loc[17.5:19.0].min() >= 0.95 * profile.loc[17.5:19.0].max()


def test_break_ratios_from_bills_match_the_real_vacation_drop(campus):
    interval, bills, hidden = campus
    full = _full_truth(interval, hidden)
    on_break = is_break(full.timestamp)
    june = full[full.timestamp.dt.month == 6]
    term = full[~on_break & full.timestamp.dt.month.isin([2, 3, 4, 8, 9, 10, 11])]
    true_ratio = june.groupby("building").kWh_consumed.mean() / term.groupby("building").kWh_consumed.mean()
    for building, ratio in ml_model.break_ratios(bills).items():
        assert abs(ratio - true_ratio[building]) < 0.1, building


def test_part_break_months_are_split_correctly(evaluation):
    """May, July and December mix term and vacation days; the lab barely slows down in vacations
    while the academic block nearly closes, so a shared vacation drop misplaces their load."""
    series, _ = evaluation
    lab = series[(series.building == "B5_Lab") & series.timestamp.dt.month.isin([5, 7, 12])].dropna(subset=["actual"])
    error = 100 * ((lab.predicted - lab.actual).abs() / lab.actual.clip(lower=0.01)).mean()
    assert error < 15


def test_generator_covers_a_year_with_quieter_breaks(campus):
    interval, bills, _ = campus
    assert bills.groupby("building").size().eq(12).all()
    academic = interval[interval.building == "B3_Academic"].dropna(subset=["kWh_consumed"])
    on_break = is_break(academic.timestamp)
    weekday_daytime = (academic.timestamp.dt.day_of_week < 5) & academic.timestamp.dt.hour.between(10, 16)
    term_load = academic.loc[weekday_daytime & ~on_break, "kWh_consumed"].mean()
    break_load = academic.loc[weekday_daytime & on_break, "kWh_consumed"].mean()
    assert break_load < 0.8 * term_load


def test_buildings_are_much_quieter_during_breaks(campus):
    interval, _, hidden = campus
    full = _full_truth(interval, hidden)
    full = full[full.timestamp.dt.month.isin([5, 6, 7, 8])]  # same season for break and term
    on_break = is_break(full.timestamp)
    ratio = full[on_break].groupby("building").kWh_consumed.mean() / full[~on_break].groupby("building").kWh_consumed.mean()
    assert ratio["B3_Academic"] < 0.45 and ratio["B4_Library"] < 0.5
    assert ratio["B1_Hostel_A"] < 0.5 and ratio["B2_Hostel_B"] < 0.5
    assert ratio["B5_Lab"] > 0.7  # research keeps running
