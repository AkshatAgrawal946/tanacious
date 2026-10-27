"""CLI pipeline: generate reconstructed demo outputs and a masked-meter metric."""
from pathlib import Path

import pandas as pd

from src import ml_model
from src.compare import compare_models
from src.data import load_data
from src.metrics import score_prediction
from src.reconstruct import reconstruct_all, reconstruct_building_month


def masked_meter_evaluation(interval: pd.DataFrame, bills: pd.DataFrame, building: str | None = None) -> dict[str, float]:
    """Treat a metered building as monthly-only without ever using its values as a reference."""
    target = building or sorted(interval["building"].unique())[0]
    truth = interval[interval["building"] == target].dropna(subset=["kWh_consumed"]).copy()
    reference = interval[interval["building"] != target].copy()
    predictions = [reconstruct_building_month(reference, bills, target, month) for month in sorted(truth["timestamp"].dt.to_period("M").astype(str).unique())]
    predicted = pd.concat(predictions, ignore_index=True)
    merged = truth.merge(predicted[["timestamp", "kWh_consumed"]], on="timestamp", suffixes=("_actual", "_predicted"))
    metrics = score_prediction(merged["kWh_consumed_actual"], merged["kWh_consumed_predicted"], merged["timestamp"])
    # The per-interval truth intentionally contains sensor gaps. Billing accuracy must therefore
    # be checked against the retained monthly bills, not against the incomplete meter sum.
    target_bills = bills[bills["building"] == target]
    metrics["bill_total_error_kwh"] = float(abs(predicted["kWh_consumed"].sum() - target_bills["monthly_kWh"].sum()))
    return metrics


def main() -> None:
    interval, bills = load_data()
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct_all(interval, bills)
    reconstructed.to_csv(output / "reconstructed_15min.csv", index=False)
    metrics = masked_meter_evaluation(interval, bills)
    pd.DataFrame([metrics]).to_csv(output / "masked_meter_metrics.csv", index=False)
    print("Wrote data/processed/reconstructed_15min.csv")
    print(pd.Series(metrics).round(3).to_string())

    model_path = ml_model.save(ml_model.train(interval, bills))
    print(f"Trained the Random Forest and saved it to {model_path}")
    # Ground truth is opened only here, to score both models after they have predicted.
    hidden = pd.read_csv(Path("data/raw") / "ground_truth_15min.csv", parse_dates=["timestamp"])
    series, comparison = compare_models(interval, bills, hidden)
    series.to_csv(output / "model_comparison_15min.csv", index=False)
    comparison.to_csv(output / "model_comparison_metrics.csv", index=False)
    print("Wrote data/processed/model_comparison_15min.csv and model_comparison_metrics.csv")
    print(comparison.pivot_table(index="building", columns="model", values="mape_pct").round(1).to_string())


if __name__ == "__main__":
    main()
