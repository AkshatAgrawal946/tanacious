"""CLI pipeline: train the Random Forest, reconstruct every building-month and check accuracy."""
from pathlib import Path

import pandas as pd

from src import ml_model
from src.data import load_data
from src.evaluate import evaluate_model
from src.reconstruct import reconstruct_all


def main() -> None:
    interval, bills = load_data()
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)

    model = ml_model.train(interval, bills)
    print(f"Trained the Random Forest and saved it to {ml_model.save(model)}")
    reconstructed = reconstruct_all(interval, bills, ml_model.shape_fn(model, interval, bills))
    reconstructed.round(3).to_csv(output / "reconstructed_15min.csv", index=False)
    print("Wrote data/processed/reconstructed_15min.csv")

    # Ground truth is opened only here, to score the model after it has predicted.
    hidden = pd.read_csv(Path("data/raw") / "ground_truth_15min.csv", parse_dates=["timestamp"])
    series, scores = evaluate_model(interval, bills, hidden)
    series.round(3).to_csv(output / "evaluation_15min.csv", index=False)
    scores.to_csv(output / "evaluation_metrics.csv", index=False)
    print("Wrote data/processed/evaluation_15min.csv and evaluation_metrics.csv")
    print(scores.set_index("building")[["evaluated_against", "mape_pct", "peak_window_error_pct"]].round(1).to_string())


if __name__ == "__main__":
    main()
