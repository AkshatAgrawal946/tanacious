"""Forecasting contracts, leakage guards, baselines, Chronos adapter (mocked), backtest and routing."""
import json

import numpy as np
import pandas as pd
import pytest

from src.forecasting import selection
from src.forecasting.backtest import build_requests, run_benchmark
from src.forecasting.baselines import RandomForestForecaster, previous_day, previous_week
from src.forecasting.chronos_adapter import ChronosForecaster, verify_checkpoint
from src.forecasting.config import ForecastExperimentConfig
from src.forecasting.data_prep import (chronos_training_inputs, chronos_validation_inputs, forecast_origins,
                                       make_request, validate_training_inputs)
from src.forecasting.interfaces import FREQ, ForecasterUnavailable, ForecastRequest, ForecastResult, ForecastTimeout, validate_result
from src.generate_data import generate


def _series(days: int = 30, start: str = "2026-03-01", missing: slice | None = None) -> pd.Series:
    index = pd.date_range(start, periods=days * 96, freq=FREQ)
    values = 2 + np.sin(2 * np.pi * np.arange(len(index)) / 96) + 0.3 * (index.day_of_week < 5)
    series = pd.Series(values, index=index)
    if missing is not None:
        series.iloc[missing] = np.nan
    return series


def _request(series=None, origin="2026-03-20", context=1344, horizon=96) -> ForecastRequest:
    return make_request(series if series is not None else _series(), "B", pd.Timestamp(origin), context, horizon)


# -- request contract / leakage --------------------------------------------------------------
def test_request_history_must_end_before_origin():
    series = _series()
    history = series[:"2026-03-20 00:00"]  # includes the origin itself
    with pytest.raises(ValueError, match="leakage guard"):
        ForecastRequest("B", pd.Timestamp("2026-03-20"), history, 96)


def test_request_rejects_unknown_future_covariates():
    request = _request()
    future = request.future_covariates.assign(temperature=25.0)
    with pytest.raises(ValueError, match="past values"):
        ForecastRequest("B", request.origin, request.history, 96, request.past_covariates, future)


def test_request_rejects_misaligned_future_covariates():
    request = _request()
    with pytest.raises(ValueError, match="exactly the forecast timestamps"):
        ForecastRequest("B", request.origin, request.history, 96, request.past_covariates, request.future_covariates.iloc[:-1])


def test_default_scenario_has_no_measured_temperature():
    request = _request()
    assert "temperature" not in request.future_covariates.columns
    assert set(request.future_covariates.columns) == {"hour", "day_of_week", "is_weekend", "is_break"}


def test_history_never_contains_future_values():
    series = _series()
    origin = pd.Timestamp("2026-03-20")
    poisoned = series.copy()
    poisoned[poisoned.index >= origin] = 1e6
    for model in (previous_day(), previous_week()):
        assert model.predict([_request(poisoned)])[0].mean.max() < 100


def test_discontinuous_history_is_regularised_with_nans():
    request = _request(_series(missing=slice(1500, 1600)))
    assert request.history.isna().sum() == 100
    assert (request.history.index.to_series().diff().iloc[1:] == FREQ).all()


# -- result validation -----------------------------------------------------------------------
def _result(request, mean=None, quantiles=None, timestamps=None):
    return ForecastResult(request.building, request.origin, timestamps if timestamps is not None else request.forecast_index,
                          np.ones(request.horizon) if mean is None else mean, "m", "v", quantiles or {})


def test_validate_result_checks_alignment_shape_and_values():
    request = _request()
    with pytest.raises(ValueError, match="misaligned"):
        validate_result(request, _result(request, timestamps=request.forecast_index + FREQ))
    with pytest.raises(ValueError, match="shape"):
        validate_result(request, _result(request, mean=np.ones(95)))
    with pytest.raises(ValueError, match="NaN"):
        validate_result(request, _result(request, mean=np.full(96, np.nan)))
    with pytest.raises(ValueError, match="cross"):
        validate_result(request, _result(request, quantiles={0.1: np.full(96, 2.0), 0.9: np.ones(96)}))
    clipped = validate_result(request, _result(request, mean=np.full(96, -1.0)))
    assert (clipped.mean == 0).all() and clipped.metadata["negative_values_clipped"] == 96


# -- baselines -------------------------------------------------------------------------------
def test_seasonal_baselines_repeat_the_right_lag():
    series = _series()
    request = _request(series)
    day = previous_day().predict([request])[0]
    week = previous_week().predict([request])[0]
    assert np.allclose(day.mean, series.reindex(request.forecast_index - pd.Timedelta(days=1)).to_numpy())
    assert np.allclose(week.mean, series.reindex(request.forecast_index - pd.Timedelta(days=7)).to_numpy())
    assert day.timestamps.equals(request.forecast_index)


def test_seasonal_baseline_falls_back_over_missing_lags():
    origin = pd.Timestamp("2026-03-20")
    series = _series()
    series[(series.index >= origin - pd.Timedelta(days=1)) & (series.index < origin)] = np.nan
    result = previous_day().predict([_request(series)])[0]
    assert np.allclose(result.mean, series.reindex(result.timestamps - pd.Timedelta(days=2)).to_numpy())


def test_random_forest_forecaster_refuses_origins_inside_training():
    series = _series(days=60)
    train = [make_request(series, "B", o, 1344, 96) for o in pd.date_range("2026-03-15", "2026-04-10", freq="D")]
    model = RandomForestForecaster(n_estimators=10).fit(train, [series.reindex(r.forecast_index) for r in train])
    with pytest.raises(ValueError, match="inside the training period"):
        model.predict([train[3]])
    result = model.predict([make_request(series, "B", pd.Timestamp("2026-04-20"), 1344, 96)])[0]
    assert result.timestamps.equals(pd.date_range("2026-04-20", periods=96, freq=FREQ))


# -- data prep: splits ------------------------------------------------------------------------
def test_training_inputs_stay_inside_their_split_and_split_on_long_gaps():
    series = _series(days=60, missing=slice(2000, 2300))  # > 1 day gap splits the series
    start, end = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-15")
    inputs = chronos_training_inputs({"B": series}, start, end, 96)
    assert len(inputs) == 2
    for item in inputs:
        assert item["start"] >= start and item["start"] + len(item["target"]) * FREQ <= end
        assert not np.isnan(item["target"]).all()
    assert validate_training_inputs(inputs, 96, split_end=end) == []
    assert validate_training_inputs(inputs, 96, split_end=pd.Timestamp("2026-03-20"))  # overlap is flagged


def test_validation_targets_lie_inside_validation_split():
    series = _series(days=60)
    start, end = pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-20")
    inputs = chronos_validation_inputs({"B": series}, start, end, 1344, 96, 96)
    assert len(inputs) == 10
    for item in inputs:
        target_start = item["start"] + 1344 * FREQ
        assert start <= target_start and target_start + 96 * FREQ <= end


def test_forecast_origins_skip_windows_with_missing_targets():
    series = _series(days=30, missing=slice(20 * 96, 21 * 96))
    origins = forecast_origins(series, pd.Timestamp("2026-03-15"), pd.Timestamp("2026-03-25"), 96, 96, 1344, 0.5)
    assert pd.Timestamp("2026-03-21") not in origins and pd.Timestamp("2026-03-20") in origins


def test_config_rejects_overlapping_splits():
    with pytest.raises(ValueError, match="non-overlapping"):
        ForecastExperimentConfig(validation=("2026-07-15", "2026-09-01"))
    with pytest.raises(ValueError, match="covariate_scenario"):
        ForecastExperimentConfig(covariate_scenario="future_bills")


# -- Chronos adapter (mocked pipeline) ---------------------------------------------------------
class FakePipeline:
    def __init__(self, delay: float = 0.0):
        self.delay, self.calls = delay, []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size):
        import time

        import torch
        time.sleep(self.delay)
        self.calls.append(inputs)
        out = []
        for item in inputs:
            level = float(np.nanmean(item["target"]))
            q = torch.tensor([[[level * (0.5 + ql) for ql in quantile_levels]] * prediction_length])
            out.append(q)
        return out, [o[..., quantile_levels.index(0.5)] for o in out]


def _chronos(pipeline, **kwargs):
    model = ChronosForecaster("amazon/chronos-2", **kwargs)
    model._pipeline, model.device = pipeline, "cpu"
    return model


def test_chronos_output_is_timestamp_aligned_and_probabilistic():
    pytest.importorskip("torch")
    requests = [_request(origin="2026-03-20"), _request(origin="2026-03-21")]
    results = _chronos(FakePipeline()).predict(requests)
    for request, result in zip(requests, results):
        assert result.timestamps.equals(request.forecast_index)
        assert set(result.quantiles) == {0.1, 0.5, 0.9}
        assert (result.quantiles[0.1] <= result.mean).all() and (result.mean <= result.quantiles[0.9]).all()
        assert np.allclose(result.mean, result.quantiles[0.5])


def test_chronos_receives_only_known_future_covariates():
    pytest.importorskip("torch")
    pipeline = FakePipeline()
    _chronos(pipeline).predict([_request()])
    item = pipeline.calls[0][0]
    assert len(item["target"]) == 1344 and set(item["future_covariates"]) <= set(item["past_covariates"])
    assert all(len(v) == 96 for v in item["future_covariates"].values())


def test_chronos_timeout_is_explicit():
    pytest.importorskip("torch")
    with pytest.raises(ForecastTimeout):
        _chronos(FakePipeline(delay=0.5), timeout_s=0.05).predict([_request()])


def test_missing_checkpoint_is_unavailable(tmp_path):
    with pytest.raises(ForecasterUnavailable):
        ChronosForecaster(str(tmp_path / "nope")).load()
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["T5Model"]}))
    with pytest.raises(ForecasterUnavailable, match="not a Chronos-2 checkpoint"):
        verify_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Chronos2Model"], "chronos_config": {"context_length": 8192}}))
    assert verify_checkpoint(tmp_path)["kind"] == "full_model"


# -- routing and fallback ----------------------------------------------------------------------
def test_routing_uses_validated_threshold_and_measured_readings_only():
    series = _series(days=30)
    summary = {"history_threshold": {"status": "measured", "min_history_steps": 672, "model": "x"}, "created_at": "t"}
    origin = pd.Timestamp("2026-03-20")
    assert selection.route(series, "B", origin, 1344, summary).mode == "forecast"
    gappy = series.copy()
    gappy[(gappy.index >= origin - pd.Timedelta(days=10)) & (gappy.index < origin)] = np.nan
    assert selection.route(gappy, "B", origin, 1344, summary).mode == "reconstruct"
    assert selection.route(None, "B", origin, 1344, summary).mode == "reconstruct"
    assert selection.min_history_steps(None)[0] == selection.DEFAULT_MIN_HISTORY_STEPS


def test_provider_failure_falls_back_to_baseline_and_says_so(monkeypatch):
    def broken(name, device="cpu", train_forest=None):
        raise ForecasterUnavailable("service down")

    monkeypatch.setattr(selection, "build_forecaster", broken)
    chosen = selection.forecast_with_fallback([_request()], "chronos2_zero_shot")
    assert chosen.used == "seasonal_previous_week" and chosen.requested == "chronos2_zero_shot"
    assert "service down" in chosen.fallback_reason
    assert chosen.results[0].model_name == "seasonal_previous_week"


def test_unregistered_finetuned_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("src.forecasting.registry.REGISTRY_PATH", tmp_path / "r.json")
    with pytest.raises(ForecasterUnavailable, match="not registered"):
        selection.build_forecaster("chronos2_finetuned_local")


# -- benchmark on generated data ---------------------------------------------------------------
@pytest.fixture(scope="module")
def campus(tmp_path_factory):
    interval, bills, _ = generate(tmp_path_factory.mktemp("raw"))
    return interval, bills


def _small_config():
    return ForecastExperimentConfig(validation=("2026-08-01", "2026-08-08"), test=("2026-09-01", "2026-09-08"),
                                    history_lengths_days=(7, 14))


def test_backtest_is_fair_repeatable_and_reports_unavailable_models(campus, monkeypatch):
    interval, bills = campus

    class Broken:
        name, version = "chronos2_zero_shot", "x"

        def predict(self, requests):
            raise ForecasterUnavailable("weights not downloaded")

    monkeypatch.setattr("src.forecasting.backtest.candidate_models", lambda config, include, finetuned: [Broken()])
    first = run_benchmark(_small_config(), interval, bills, output_dir=None)
    second = run_benchmark(_small_config(), interval, bills, output_dir=None)
    pd.testing.assert_frame_equal(first["scores"], second["scores"])
    summary = first["summary"]
    assert {(r["model"], r["status"]) for r in summary["runs"] if r["model"] == "chronos2_zero_shot"} == {("chronos2_zero_shot", "not_run")}
    assert summary["gates"]["chronos2_zero_shot"]["promoted"] is False
    assert summary["default_model"] == summary["strongest_simple_baseline"]
    predictions = first["predictions"]
    slices = predictions.groupby(["split", "model"]).apply(lambda g: len(g), include_groups=False).unstack()
    assert (slices.nunique(axis=1) == 1).all()  # every model scored on identical slices
    assert "chronos2_zero_shot" not in set(predictions["model"])  # no silent substitution


def test_rf_forecaster_trains_only_before_validation(campus):
    interval, _ = campus
    config = _small_config()
    pairs = build_requests(config, interval, "train")
    assert max(r.forecast_index[-1] for r, _ in pairs) < config.split("validation")[0]
    assert all(r.history.index[-1] < r.origin for r, _ in pairs)


def test_gate_requires_material_improvement(campus):
    from src.forecasting.backtest import promotion_decision

    config = ForecastExperimentConfig()
    scores = pd.DataFrame([
        {"split": "test", "building": "ALL", "model": "cand", "wape_pct": 9.5, "peak_window_energy_error_pct": 1.0, "peak_timing_error_min": 30},
        {"split": "test", "building": "ALL", "model": "base", "wape_pct": 10.0, "peak_window_energy_error_pct": 1.0, "peak_timing_error_min": 30},
    ])
    runs = [{"model": "cand", "split": "test", "status": "measured", "latency_s_per_forecast": 0.1}]
    assert promotion_decision(config, scores, "cand", "base", runs)["promoted"] is False  # 5% < 10%
    scores.loc[0, "wape_pct"] = 8.0
    assert promotion_decision(config, scores, "cand", "base", runs)["promoted"] is True
    scores.loc[0, "peak_window_energy_error_pct"] = 5.0
    assert promotion_decision(config, scores, "cand", "base", runs)["promoted"] is False  # peak regression
