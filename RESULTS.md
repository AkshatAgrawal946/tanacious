# Benchmark results (synthetic data, 2026-10-29)

Every number below was measured on the synthetic 2026 dataset (`dataset_fingerprint efad85610ab87cff`) on a CPU-only Mac
(Python 3.14.7, torch 2.15.0, chronos-forecasting 2.3.2, scikit-learn 1.9.1). They are not real-campus accuracy.
Full provenance is in `data/processed/forecast_benchmark/summary.json` and `data/processed/evaluation_metrics.meta.json`.

## Task B: forecasting (24 h ahead, 14-day context, daily midnight origins)

B3_Academic + B5_Lab; validation = Aug 2026 (31 origins), test = Sep 2026 (30 origins, 5,760 scored intervals per model).
All models were scored on identical intervals.

| Model | Val WAPE % | Test WAPE % | Test MAE kWh | Test RMSE kWh | 4-8 PM energy err % | Peak timing err (min) | P10-P90 coverage | Latency / forecast |
|---|---|---|---|---|---|---|---|---|
| seasonal_previous_day | 9.66 | 9.17 | 1.15 | 1.58 | 0.09 | 72.5 | – | ~1 ms |
| seasonal_previous_week | **7.26** (strongest baseline) | 7.33 | 0.92 | 1.17 | 0.59 | 71.0 | – | ~1 ms |
| random_forest_forecast | 9.17 | 6.06 | 0.76 | 0.98 | 3.03 | 45.0 | – | ~17 ms |
| chronos2_zero_shot | 5.28 | 5.50 | 0.69 | 0.88 | 0.15 | 51.5 | 0.78 | ~80 ms |
| chronos2_finetuned_local (LoRA, 300 steps) | 5.24 | 5.44 | 0.68 | 0.87 | 0.06 | 49.5 | 0.78 | ~90 ms |

Promotion gate (vs. `seasonal_previous_week`, test month): zero-shot −25.1% relative WAPE, fine-tuned −25.9%; neither
regresses peak energy, peak timing or latency, so both pass. The default is the one with the lower **validation** WAPE
(fine-tuned). The zero-shot/fine-tuned gap (0.06 WAPE points on 60 forecasts) is too small to be meaningful.

Per building (test WAPE %): B3_Academic: fine-tuned 7.03, zero-shot 7.09, RF 7.89, prev-week 9.56, prev-day 13.20;
B5_Lab: fine-tuned 4.47, zero-shot 4.54, RF 4.95, prev-week 5.99, prev-day 6.73.

History threshold (validation, default model): 1 day 15.4%, 2 days 9.1%, 3 days 8.1%, 7 days 5.8%, 14 days 5.2% WAPE,
so routing requires 14 days (1,344 measured readings) before a building is forecast.

The fine-tuned checkpoint lives in `models/checkpoints/` (not committed). On a machine without it, the dashboard's
Automatic choice falls back to the previous-week baseline and says so; re-run the fine-tune or pick `chronos2_zero_shot`.

Fine-tuning run: LoRA (1.2 M of 120.7 M parameters trainable), lr 1e-5, batch 32, 300 steps, 821 s on CPU; validation
loss 0.5805 → 0.5784 (best checkpoint selected on validation loss).

## Task A: reconstruction

Building-level holdouts (`python -m src.pipeline`); WAPE / MAPE % on reconstructed intervals:

| Building | Scenario | WAPE | MAPE | 4-8 PM energy err % |
|---|---|---|---|---|
| B1_Hostel_A | hidden truth (outside pilot weeks) | 5.1 | 6.0 | 1.3 |
| B2_Hostel_B | hidden truth | 5.5 | 6.6 | 0.5 |
| B3_Academic | unmetered (own meter held out) | 11.3 | 14.5 | 0.2 |
| B4_Library | hidden truth (outside pilot weeks) | 7.5 | 9.5 | 0.9 |
| B5_Lab | unmetered (own meter held out) | 7.2 | 7.2 | 4.6 |

Partially metered (WAPE % on hidden days; RF vs flat bill split):

| Building | 0% metered | 25% | 50% | 75% |
|---|---|---|---|---|
| B3_Academic | 11.3 vs 29.2 | 7.5 vs 28.6 | 7.3 vs 28.3 | 7.0 vs 27.7 |
| B5_Lab | 7.2 vs 11.8 | 4.0 vs 11.8 | 3.9 vs 11.6 | 3.9 vs 11.7 |

Observed readings were returned unchanged in every scenario (max absolute change 0 kWh).

## Not run / unverified

- **Adaption Labs fine-tuning: not available for this account.** A live, read-only `adaption-check` on 2026-10-29
  authenticated successfully (`GET /api/v1/training-models` → 200) but the account offers no Chronos training model.
  No job was submitted, no data uploaded. The local LoRA fine-tune above is the only fine-tuned Chronos-2.
- Live tests (`tests/test_live_integration.py`) were skipped in the default run (opt-in).
- GPU inference/training was not tested.

## Reproduce

```bash
pip install -r requirements-forecast.txt
python -m src.generate_data && python -m src.pipeline
python -m src.forecasting.finetune local --steps 300 --batch-size 32 --learning-rate 1e-5
python -m src.forecasting.backtest
```
