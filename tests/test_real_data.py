import numpy as np
import pandas as pd

from src.real_data import clean_meter


def _day(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2017-03-01", periods=len(values), freq="h"), dtype=float)


def test_glitches_are_blanked_and_counted():
    base = 100 + 20 * np.sin(np.arange(72) / 24 * 2 * np.pi)
    values = base.copy()
    values[10] = -5  # negative
    values[30] = 900  # one-hour spike
    values[50] = 3  # dropout
    values[60:67] = 77.0  # frozen meter
    values[5] = np.nan
    cleaned, counts = clean_meter(_day(values))
    assert counts == {"missing": 1, "negative": 1, "stuck": 7, "spike": 1, "dropout": 1}
    assert cleaned.isna().sum() == 11
    kept = cleaned.notna()
    assert np.array_equal(cleaned[kept], _day(values)[kept])  # nothing else is changed


def test_real_patterns_are_kept():
    hours = np.arange(24 * 7)
    library = np.where(np.isin(hours % 24, [4, 5]), 40.0, 190.0)  # a scheduled 4-5 AM shutdown every day
    library[24 * 5 + 13:24 * 5 + 16] = 400.0  # a three-hour opening on a quiet day
    library = library + np.arange(len(library)) % 3  # real meters never repeat one value for hours
    cleaned, counts = clean_meter(_day(library))
    assert counts["spike"] == counts["dropout"] == counts["stuck"] == 0 and cleaned.notna().all()
