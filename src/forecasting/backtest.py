"""Rolling-origin forecasting benchmark (Task B).

    python -m src.forecasting.backtest                 # baselines + zero-shot Chronos-2 (+ registered fine-tuned models)
    python -m src.forecasting.backtest --skip-chronos  # baselines only

Protocol:
* Daily midnight origins; each forecast is ``horizon_steps`` ahead from ``context_steps`` of measured
  history. Every model receives the *same* requests and is scored on the *same* intervals (measured
  actuals only), per split.
* The Random Forest forecaster trains on origins inside the training split only.
* The strongest simple baseline is chosen on the validation split; the promotion gate compares the
  candidate with it on the test split, which is used for nothing else.
* A history-length experiment on the validation split gives the minimum history for which
  forecasting is trusted (used by ``selection.py`` to route between forecasting and reconstruction).
* Models that cannot run are reported as ``not_run`` with the reason; they are never replaced by a
  baseline inside the benchmark.

Artifacts (``data/processed/forecast_benchmark/``): ``interval_predictions.csv`` (every forecast
interval), ``scores.csv`` (aggregates) and ``summary.json`` (config, data fingerprint, environment,
gate decision, history threshold).
"""
import argparse
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import ml_model, provenance
from src.data import load_data
from src.forecasting import registry
from src.forecasting.baselines import RandomForestForecaster, previous_day, previous_week
from src.forecasting.chronos_adapter import ChronosForecaster
from src.forecasting.config import ForecastExperimentConfig
from src.forecasting.data_prep import actuals, building_series, forecast_origins, make_request
from src.forecasting.interfaces import ForecasterUnavailable
from src.metrics import forecast_scores

log = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/processed/forecast_benchmark")
CONFIG_PATH = Path("configs/forecast_benchmark.json")
SIMPLE_BASELINES = ("seasonal_previous_day", "seasonal_previous_week", "random_forest_forecast")


def build_requests(config: ForecastExperimentConfig, interval: pd.DataFrame, split: str,
                   context_steps: int | None = None) -> list[tuple]:
    """(request, measured actuals) for every building and origin in a split."""
    start, end = config.split(split)
    temperature = ml_model.campus_temperature(interval)
    context = context_steps or config.context_steps
    pairs = []
    for building in config.buildings:
        series = building_series(interval, building)
        for origin in forecast_origins(series, start, end, config.horizon_steps, config.origin_stride_steps,
                                       config.context_steps, config.max_missing_target_fraction):
            request = make_request(series, building, origin, context, config.horizon_steps,
                                   config.covariate_scenario, temperature)
            pairs.append((request, actuals(series, origin, config.horizon_steps)))
    return pairs


def fit_random_forest(config: ForecastExperimentConfig, interval: pd.DataFrame) -> RandomForestForecaster:
    pairs = build_requests(config, interval, "train")
    return RandomForestForecaster(seed=config.seed).fit([r for r, _ in pairs], [a for _, a in pairs])


def run_model(model, pairs: list[tuple], split: str) -> tuple[pd.DataFrame | None, dict]:
    """Forecast every request; returns (interval predictions, run info). Failures are reported."""
    start = time.perf_counter()
    try:
        results = model.predict([r for r, _ in pairs])
    except (ForecasterUnavailable, ImportError, RuntimeError) as error:
        log.warning("%s could not run on %s: %s", model.name, split, error)
        return None, {"model": model.name, "split": split, "status": "not_run", "reason": str(error)}
    frames = []
    for (request, actual), result in zip(pairs, results):
        frame = result.to_frame().assign(split=split, step=np.arange(request.horizon), actual=actual.to_numpy())
        frames.append(frame)
    elapsed = time.perf_counter() - start
    # ru_maxrss is bytes on macOS and KiB on Linux; it is the process peak, not this model's alone.
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return pd.concat(frames, ignore_index=True), {
        "model": model.name, "model_version": model.version, "split": split, "status": "measured",
        "forecasts": len(results), "wall_time_s": elapsed, "latency_s_per_forecast": elapsed / max(len(results), 1),
        "process_peak_rss_mb": peak_rss_mb}


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["split", "model", "model_version"]):
        for building, part in [("ALL", group), *group.groupby("building")]:
            lower, upper = (part["q10"], part["q90"]) if {"q10", "q90"} <= set(part.columns) and part["q10"].notna().any() else (None, None)
            rows.append({"split": keys[0], "model": keys[1], "model_version": keys[2], "building": building,
                         **forecast_scores(part["actual"], part["predicted"], part["timestamp"], lower, upper)})
    return pd.DataFrame(rows)


def promotion_decision(config: ForecastExperimentConfig, scores: pd.DataFrame, candidate: str, baseline: str,
                       runs: list[dict]) -> dict:
    """Configurable gate: >= X% relative WAPE improvement on the test split over the strongest simple
    baseline, without material peak regression or unacceptable latency."""
    test = scores[(scores["split"] == "test") & (scores["building"] == "ALL")].set_index("model")
    if candidate not in test.index or baseline not in test.index:
        return {"candidate": candidate, "baseline": baseline, "promoted": False, "reason": "candidate or baseline was not measured"}
    c, b = test.loc[candidate], test.loc[baseline]
    improvement = (b["wape_pct"] - c["wape_pct"]) / b["wape_pct"]
    latency = next((r["latency_s_per_forecast"] for r in runs if r["model"] == candidate and r["split"] == "test"
                    and r["status"] == "measured"), float("inf"))
    checks = {
        "relative_wape_improvement": float(improvement),
        "wape_ok": bool(improvement >= config.min_relative_wape_improvement),
        "peak_energy_ok": bool(c["peak_window_energy_error_pct"] <= b["peak_window_energy_error_pct"]
                               + config.max_peak_energy_error_regression_pct_points),
        "peak_timing_ok": bool(c["peak_timing_error_min"] <= b["peak_timing_error_min"] + config.max_peak_timing_regression_min),
        "latency_ok": bool(latency <= config.max_latency_s_per_forecast),
        "latency_s_per_forecast": float(latency),
    }
    promoted = all(v for k, v in checks.items() if k.endswith("_ok"))
    return {"candidate": candidate, "baseline": baseline, "promoted": promoted, "checks": checks,
            "default_model": candidate if promoted else baseline}


def history_threshold(config: ForecastExperimentConfig, interval: pd.DataFrame, model) -> dict:
    """Validation WAPE by history length; the threshold is the shortest history whose WAPE is within
    5% (relative) of the full-context WAPE."""
    rows = []
    for days in sorted(config.history_lengths_days):
        pairs = build_requests(config, interval, "validation", context_steps=days * 96)
        predictions, info = run_model(model, pairs, "validation")
        if predictions is None:
            return {"model": model.name, "status": "not_run", "reason": info["reason"]}
        rows.append({"history_days": days, "wape_pct": forecast_scores(predictions["actual"], predictions["predicted"],
                                                                         predictions["timestamp"])["wape_pct"]})
    full = rows[-1]["wape_pct"]
    threshold = next(r["history_days"] for r in rows if r["wape_pct"] <= 1.05 * full)
    return {"model": model.name, "status": "measured", "by_history_days": rows, "min_history_days": threshold,
            "min_history_steps": threshold * 96, "rule": "shortest history with validation WAPE <= 1.05 x full-context WAPE"}


def candidate_models(config: ForecastExperimentConfig, include_chronos: bool, finetuned: list[str]) -> list:
    models = []
    if include_chronos:
        models.append(ChronosForecaster(config.chronos_model_id, device=config.device, quantile_levels=config.quantile_levels,
                                        timeout_s=config.inference_timeout_s, seed=config.seed))
    for name in finetuned:
        record = registry.get(name)
        ok, reason = registry.usable(record)
        if not ok:
            log.warning("fine-tuned model %s is not usable: %s", name, reason)
            models.append(_Unavailable(name, reason))
            continue
        models.append(ChronosForecaster(record.location, name=name, version=record.version, device=config.device,
                                        quantile_levels=config.quantile_levels, timeout_s=config.inference_timeout_s,
                                        seed=config.seed))
    return models


class _Unavailable:
    """Placeholder so a model that cannot run still appears in the report as not_run."""

    def __init__(self, name: str, reason: str):
        self.name, self.version, self.reason = name, "unavailable", reason

    def predict(self, requests):
        raise ForecasterUnavailable(self.reason)


def run_benchmark(config: ForecastExperimentConfig, interval: pd.DataFrame, bills: pd.DataFrame,
                  include_chronos: bool = True, finetuned: list[str] | None = None,
                  output_dir: Path | None = OUTPUT_DIR) -> dict:
    fingerprint = provenance.dataset_fingerprint(interval, bills)
    forest = fit_random_forest(config, interval)
    models = [previous_day(), previous_week(), forest, *candidate_models(config, include_chronos, finetuned or [])]
    frames, runs = [], []
    for split in ("validation", "test"):
        pairs = build_requests(config, interval, split)
        for model in models:
            predictions, info = run_model(model, pairs, split)
            runs.append(info)
            if predictions is not None:
                frames.append(predictions)
    predictions = pd.concat(frames, ignore_index=True)
    # Identical evaluation slices: every measured model must cover the same (split, building, timestamp) set.
    coverage = predictions.groupby(["split", "model"]).apply(lambda g: frozenset(zip(g["building"], g["timestamp"])),
                                                              include_groups=False)
    for split, sets in coverage.groupby(level="split"):
        if len(set(sets)) != 1:
            raise AssertionError(f"models were evaluated on different intervals in the {split} split")
    scores = score(predictions)

    validation = scores[(scores["split"] == "validation") & (scores["building"] == "ALL")].set_index("model")
    strongest = validation.loc[[m for m in SIMPLE_BASELINES if m in validation.index], "wape_pct"].idxmin()
    measured = set(validation.index)
    candidates = [m.name for m in models if m.name not in SIMPLE_BASELINES]
    gates = {name: promotion_decision(config, scores, name, strongest, runs) if name in measured
             else {"candidate": name, "promoted": False, "reason": next(r["reason"] for r in runs if r["model"] == name)}
             for name in candidates}
    promoted = [g["candidate"] for g in gates.values() if g.get("promoted")]
    # Among promoted candidates, pick by validation WAPE: the test month only decides pass/fail.
    default_model = min(promoted, key=lambda m: validation.loc[m, "wape_pct"]) if promoted else strongest
    routing_model = next((m for m in models if m.name == default_model), forest)
    threshold = history_threshold(config, interval, routing_model)

    summary = {
        "task": "forecasting", "created_at": provenance.now_utc(), "config": config.to_dict(),
        "config_hash": provenance.config_hash(config.to_dict()), "dataset_fingerprint": fingerprint,
        "environment": provenance.environment(), "runs": runs, "strongest_simple_baseline": strongest,
        "selection_split": "validation", "gate_split": "test", "gates": gates, "default_model": default_model,
        "history_threshold": threshold, "rf_training_cutoff": str(forest.training_cutoff),
        "requests_per_split": predictions.groupby("split")["origin"].nunique().to_dict(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.round({c: 4 for c in predictions.select_dtypes("number").columns}).to_csv(
            output_dir / "interval_predictions.csv", index=False)
        scores.to_csv(output_dir / "scores.csv", index=False)
        provenance.write_json(output_dir / "summary.json", summary)
    return {"summary": summary, "scores": scores, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--skip-chronos", action="store_true", help="baselines only (no torch / model download)")
    parser.add_argument("--finetuned", nargs="*", default=None,
                        help="registry names of fine-tuned Chronos-2 models (default: every verified local checkpoint)")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ForecastExperimentConfig.from_json(args.config) if Path(args.config).exists() else ForecastExperimentConfig()
    finetuned = args.finetuned if args.finetuned is not None else [
        name for name, record in registry.load().items() if record.kind == "local_checkpoint" and record.status == "verified"]
    interval, bills = load_data()
    result = run_benchmark(config, interval, bills, include_chronos=not args.skip_chronos, finetuned=finetuned,
                           output_dir=Path(args.output))
    summary, scores = result["summary"], result["scores"]
    columns = ["wape_pct", "mae_kwh", "rmse_kwh", "peak_window_energy_error_pct", "peak_timing_error_min", "p10_p90_coverage"]
    print(scores[scores["building"] == "ALL"].set_index(["split", "model"])[[c for c in columns if c in scores]].round(2).to_string())
    for run in summary["runs"]:
        if run["status"] != "measured":
            print(f"NOT RUN: {run['model']} ({run['split']}): {run['reason']}")
    print(f"strongest simple baseline (validation): {summary['strongest_simple_baseline']}")
    for gate in summary["gates"].values():
        print(f"gate {gate['candidate']}: promoted={gate['promoted']} {gate.get('checks', gate.get('reason'))}")
    print(f"default forecasting model: {summary['default_model']}; min history: {summary['history_threshold'].get('min_history_days')} days")


if __name__ == "__main__":
    main()
