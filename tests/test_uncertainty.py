import numpy as np
import pandas as pd

from src.uncertainty import CAMPUS, calibrate, likely_range


def _series(ratio_by_hour, is_pilot=False) -> pd.DataFrame:
    """Two months of 15-minute test results for one building: actual = predicted * ratio(hour)."""
    ts = pd.date_range("2026-03-01", "2026-04-30 23:45", freq="15min")
    ratio = ts.hour.map(ratio_by_hour).to_numpy(dtype=float)
    return pd.DataFrame({"timestamp": ts, "building": "B1", "is_pilot": is_pilot,
                         "predicted": 10.0, "actual": 10.0 * ratio})


def test_ranges_contain_the_shown_value_even_for_a_one_sided_bias():
    cal = calibrate(_series(lambda h: 0.9))  # always 10% over-predicted
    assert (cal.interval["low"] <= 1).all() and (cal.interval["high"] >= 1).all()
    assert np.isclose(cal.interval.loc[("B1", 3), "low"], 0.9)
    assert np.isclose(cal.peak_energy.loc["B1", "low"], 0.9)


def test_spread_is_per_hour_and_campus_is_calibrated_too():
    cal = calibrate(_series(lambda h: 1.2 if h == 19 else 1.0))
    assert np.isclose(cal.interval.loc[("B1", 19), "high"], 1.2)
    assert np.isclose(cal.interval.loc[("B1", 3), "high"], 1.0)
    assert (CAMPUS, 19) in cal.interval.index


def test_pilot_intervals_are_left_out():
    series = pd.concat([_series(lambda h: 1.0), _series(lambda h: 2.0, is_pilot=True)])
    assert np.isclose(calibrate(series).interval["high"].max(), 1.0)


def test_only_imputed_load_is_scaled_and_unknown_keys_give_nan():
    cal = calibrate(_series(lambda h: 0.9 if h < 12 else 1.1))
    low, high = likely_range(cal.interval, [100.0, 0.0, 100.0], [0.0, 50.0, 0.0], ["B1", "B1", "B9"], [3, 3, 3])
    assert np.allclose(low[:2], [90.0, 50.0]) and np.allclose(high[:2], [100.0, 50.0])
    assert np.isnan(low[2]) and np.isnan(high[2])
    assert np.isnan(likely_range(None, 100.0, 0.0, "B1", 3)[0])
