"""Live integration tests. Skipped unless explicitly enabled, so ordinary runs never touch the network
or download model weights. Skips here mean UNVERIFIED, not passed.

    TANACIOUS_LIVE_CHRONOS=1 pytest tests/test_live_integration.py    # real Chronos-2 weights (download)
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.forecasting.data_prep import make_request

live_chronos = pytest.mark.skipif(os.environ.get("TANACIOUS_LIVE_CHRONOS") != "1", reason="set TANACIOUS_LIVE_CHRONOS=1")


@live_chronos
def test_real_chronos2_zero_shot_forecast():
    from src.forecasting.chronos_adapter import ChronosForecaster

    index = pd.date_range("2026-03-01", periods=20 * 96, freq="15min")
    series = pd.Series(2 + np.sin(2 * np.pi * np.arange(len(index)) / 96), index=index)
    request = make_request(series, "B", pd.Timestamp("2026-03-19"), 1344, 96)
    result = ChronosForecaster().predict([request])[0]
    assert result.timestamps.equals(request.forecast_index) and np.isfinite(result.mean).all()
    assert np.mean(np.abs(result.mean - series.reindex(request.forecast_index).to_numpy())) < 0.2

