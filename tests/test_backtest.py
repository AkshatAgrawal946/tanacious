import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.backtest import (
    KW,
    KWH,
    REGISTER,
    guess_columns,
    hide_between,
    hide_hours,
    hide_random_days,
    prepare,
    rebuild,
    score,
)
from src.ml_model import type_schedule


def _academic_kwh(ts: pd.DatetimeIndex, seed: int = 0) -> np.ndarray:
    """An academic block: the type's schedule, quiet weekends, a little noise (kWh per 15 minutes)."""
    hour = ts.hour + ts.minute / 60
    weekday = np.where(ts.day_of_week < 5, 1.0, 0.3)
    return 10 * type_schedule("academic", np.asarray(hour)) * weekday * np.random.default_rng(seed).uniform(0.95, 1.05, len(ts))


def _wide(start="2025-03-01", end="2025-04-30 23:45", freq="15min") -> pd.DataFrame:
    ts = pd.date_range(start, end, freq=freq)
    return pd.DataFrame({"Time": ts.strftime("%Y-%m-%d %H:%M"), "Block A (kWh)": _academic_kwh(ts),
                         "Outdoor temp": 25.0})


def test_columns_are_guessed_for_a_typical_export():
    guess = guess_columns(_wide())
    assert guess["timestamp"] == "Time" and guess["temperature"] == "Outdoor temp"
    assert guess["meters"] == ["Block A (kWh)"] and not guess["dayfirst"]
    assert guess_columns(pd.DataFrame({"t": ["25/03/2026 10:00", "26/03/2026 10:00"], "kWh": [1, 2]}))["dayfirst"]


def test_a_clean_15_minute_file_comes_through_unchanged():
    raw = _wide()
    data = prepare(raw, "Time", ["Block A (kWh)"], KWH, temperature="Outdoor temp")
    assert data.resolution == pd.Timedelta("15min")
    assert np.allclose(data.readings["kWh_consumed"], raw["Block A (kWh)"])
    assert (data.readings["temperature"] == 25.0).all() and not data.readings["is_filled"].any()


def test_hourly_kw_becomes_kwh_spread_over_four_slots():
    raw = pd.DataFrame({"t": pd.date_range("2025-03-01", "2025-03-31 23:00", freq="h"), "kw": 8.0})
    data = prepare(raw, "t", ["kw"], KW)
    assert data.resolution == pd.Timedelta("1h")
    assert len(data.readings) == 31 * 96 and np.allclose(data.readings["kWh_consumed"], 2.0)
    assert any("No temperature" in note for note in data.notes)


def test_five_minute_register_readings_are_differenced_and_resets_dropped():
    ts = pd.date_range("2025-03-01", "2025-04-01", freq="5min")  # one reading past the month for the last diff
    register = np.arange(len(ts)) * 0.5
    register[100:] -= 10  # a meter reset
    data = prepare(pd.DataFrame({"t": ts, "reg": register}), "t", ["reg"], REGISTER)
    assert any("negative" in note for note in data.notes)
    month = data.readings[data.readings["timestamp"] < "2025-04-01"]
    assert np.allclose(month["kWh_consumed"], 1.5)  # three 0.5 kWh steps per 15 minutes; the reset slot filled
    assert month["is_filled"].sum() == 1


def test_sparse_months_are_left_out_and_small_gaps_filled():
    raw = _wide(start="2025-03-01", end="2025-05-10")  # May is a partial month
    raw = raw.drop(index=range(200, 210))  # a short dropout in March
    data = prepare(raw, "Time", ["Block A (kWh)"], KWH)
    months = data.readings["timestamp"].dt.to_period("M").astype(str).unique()
    assert list(months) == ["2025-03", "2025-04"] and any("May 2025" in note for note in data.notes)
    assert data.readings["is_filled"].sum() == 10


def test_long_format_and_timezones_and_bad_intervals():
    ts = pd.date_range("2025-03-01", "2025-03-31 23:45", freq="15min", tz="Asia/Kolkata")
    raw = pd.concat([pd.DataFrame({"when": ts.astype(str), "meter": m, "kwh": 1.0}) for m in ("North", "South")])
    data = prepare(raw, "when", ["kwh"], KWH, building="meter")
    assert set(data.readings["building"]) == {"North", "South"}
    assert data.readings["timestamp"].min() == pd.Timestamp("2025-03-01 00:00")  # wall-clock time kept
    with pytest.raises(ValueError, match="divide 15 minutes"):
        prepare(pd.DataFrame({"t": pd.date_range("2025-03-01", periods=500, freq="7min"), "v": 1.0}), "t", ["v"])
    with pytest.raises(ValueError, match="exactly one value column"):
        prepare(raw, "when", ["kwh", "kwh"], KWH, building="meter")


def test_hiding_modes():
    ts = pd.Series(pd.date_range("2025-03-01", "2025-03-10 23:45", freq="15min"))
    assert hide_between(ts, "2025-03-02", "2025-03-03").sum() == 96
    assert hide_random_days(ts, 0.3).sum() == 3 * 96
    hours = hide_hours(ts, 16, 20)
    assert hours.sum() == 10 * 16 and set(ts[hours].dt.hour) == {16, 17, 18, 19}


def test_rebuild_keeps_visible_readings_matches_totals_and_beats_the_flat_split():
    data = prepare(_wide(), "Time", ["Block A (kWh)"], KWH, temperature="Outdoor temp")
    ts = data.readings["timestamp"]
    hidden = hide_random_days(ts, 0.3)
    series, trained_on = rebuild(data, {"Block A (kWh)": "academic"}, "Block A (kWh)", hidden)
    assert "visible" in trained_on
    visible = ~series["hidden"]
    assert np.array_equal(series.loc[visible, "predicted"], series.loc[visible, "actual"])
    monthly = series.groupby(series.index.to_period("M"))[["actual", "predicted", "flat"]].sum()
    assert np.allclose(monthly["predicted"], monthly["actual"]) and np.allclose(monthly["flat"], monthly["actual"])
    scores = score(series)
    assert scores.loc["model", "scored_periods"] == hidden.sum()
    assert scores.loc["model", "wape_pct"] < scores.loc["flat_bill_split", "wape_pct"] / 2


def test_hourly_files_are_scored_per_hour_and_everything_hidden_uses_the_fallback():
    hourly = _wide(freq="h").assign(**{"Block A (kWh)": lambda f: f["Block A (kWh)"] * 4})
    data = prepare(hourly, "Time", ["Block A (kWh)"], KWH)
    ts = data.readings["timestamp"]
    series, _ = rebuild(data, {"Block A (kWh)": "academic"}, "Block A (kWh)", hide_between(ts, "2025-03-10", "2025-03-12"))
    assert score(series, data.resolution).loc["model", "scored_periods"] == 48
    with pytest.raises(ValueError, match="no fallback"):
        rebuild(data, {"Block A (kWh)": "academic"}, "Block A (kWh)", np.ones(len(ts), dtype=bool))
    flat = lambda timestamps, building, building_type: np.ones(len(timestamps))
    _, trained_on = rebuild(data, {"Block A (kWh)": "academic"}, "Block A (kWh)", np.ones(len(ts), dtype=bool), flat)
    assert "campus" in trained_on
    with pytest.raises(ValueError, match="Nothing hidden"):
        score(rebuild(data, {"Block A (kWh)": "academic"}, "Block A (kWh)", np.zeros(len(ts), dtype=bool))[0])


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="unit must be"):
        prepare(_wide(), "Time", ["Block A (kWh)"], "MWh")
    assert backtest.MIN_COVERAGE < 1


def test_decimal_and_thousands_commas_are_read_as_numbers():
    assert backtest.numbers(pd.Series(["1,5", "1.234,25", "2"])).tolist() == [1.5, 1234.25, 2.0]
    assert backtest.numbers(pd.Series(["1,234.5", "12", None])).iloc[:2].tolist() == [1234.5, 12.0]
    ts = pd.date_range("2025-03-01", "2025-03-31 23:45", freq="15min")
    raw = pd.DataFrame({"Zeit": ts.strftime("%d.%m.%Y %H:%M"), "Verbrauch": ["0,75"] * len(ts)})
    guess = guess_columns(raw)
    assert guess["meters"] == ["Verbrauch"] and guess["dayfirst"]
    data = prepare(raw, "Zeit", ["Verbrauch"], KWH, dayfirst=True)
    assert np.allclose(data.readings["kWh_consumed"], 0.75) and len(data.readings) == 31 * 96


def test_building_types_are_guessed_from_names_and_others_left_out():
    guesses = {name: backtest.guess_type(name) for name in (
        "Panther_lodging_Cora", "Fox_education_Andre", "Classroom Nilda kWh", "Research Lab kWh", "Library Bart kWh",
        "Dormitory Helen kWh", "HostelBlockA", "Panther_parking_Lorriane", "Panther_office_Hannah", "Panther_retail_Felix",
        "President's House")}
    assert guesses == {"Panther_lodging_Cora": "hostel", "Fox_education_Andre": "academic", "Classroom Nilda kWh": "academic",
                       "Research Lab kWh": "lab", "Library Bart kWh": "library", "Dormitory Helen kWh": "hostel",
                       "HostelBlockA": "hostel", "Panther_parking_Lorriane": None, "Panther_office_Hannah": None,
                       "Panther_retail_Felix": None, "President's House": None}


def test_keep_drops_left_out_buildings_in_long_files():
    ts = pd.date_range("2025-03-01", "2025-03-31 23:45", freq="15min")
    raw = pd.concat([pd.DataFrame({"t": ts, "site": m, "kwh": 1.0}) for m in ("Hostel A", "Car park")])
    data = prepare(raw, "t", ["kwh"], KWH, building="site", keep=["Hostel A"])
    assert set(data.readings["building"]) == {"Hostel A"}
