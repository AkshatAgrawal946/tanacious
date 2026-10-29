"""CLI pipeline: train the Random Forest, reconstruct every building-month and check accuracy.

``python -m src.pipeline``
"""
import logging
from pathlib import Path

from src import ml_model, provenance
from src.data import load_data
from src.evaluate import (
    evaluate_model,
    evaluate_partial_metering,
    load_ground_truth,
    observed_preservation_error,
)
from src.reconstruct import reconstruct_all

PROCESSED = Path("data/processed")


def _rounded(frame, digits: int = 3):
    return frame.round({c: digits for c in frame.select_dtypes("number").columns})
EVALUATION_META = PROCESSED / "evaluation_metrics.meta.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    interval, bills = load_data()
    fingerprint = provenance.dataset_fingerprint(interval, bills)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    model = ml_model.train(interval, bills)
    print(f"Trained the Random Forest and saved it to {ml_model.save(model, dataset_fingerprint=fingerprint)}")
    reconstructed = reconstruct_all(interval, bills, ml_model.shape_fn(model, interval, bills))
    _rounded(reconstructed).to_csv(PROCESSED / "reconstructed_15min.csv", index=False)
    print("Wrote data/processed/reconstructed_15min.csv")

    # Ground truth is opened only here, to score the model after it has predicted.
    hidden = load_ground_truth()
    series, scores = evaluate_model(interval, bills, hidden)
    preserved = observed_preservation_error(series)
    if preserved > 0:
        raise AssertionError(f"reconstruction changed an observed reading by {preserved} kWh")
    _rounded(series).to_csv(PROCESSED / "evaluation_15min.csv", index=False)
    scores.to_csv(PROCESSED / "evaluation_metrics.csv", index=False)
    partial = evaluate_partial_metering(interval, bills)
    partial.to_csv(PROCESSED / "reconstruction_partial_metering.csv", index=False)
    provenance.write_json(EVALUATION_META, {
        "task": "reconstruction", "dataset_fingerprint": fingerprint, "model_version": ml_model.MODEL_VERSION,
        "seed": ml_model.SEED, "created_at": provenance.now_utc(), "environment": provenance.environment(),
        "observed_max_abs_change_kwh": preserved,
        "artifacts": ["evaluation_15min.csv", "evaluation_metrics.csv", "reconstruction_partial_metering.csv"],
    })
    print("Wrote evaluation_15min.csv, evaluation_metrics.csv and reconstruction_partial_metering.csv")
    print(scores.set_index("building")[["evaluated_against", "wape_pct", "mape_pct", "peak_window_error_pct"]].round(1).to_string())
    print(partial.pivot_table(index=["building", "observed_fraction"], columns="model", values="wape_pct").round(1).to_string())


if __name__ == "__main__":
    main()
