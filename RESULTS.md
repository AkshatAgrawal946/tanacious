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
| **chronos2_zero_shot** | 5.28 | **5.50** | 0.69 | 0.88 | 0.15 | 51.5 | 0.78 | ~75 ms |

Promotion gate (vs. `seasonal_previous_week`, test month): zero-shot Chronos-2 is 25.1% better in relative WAPE and
does not regress peak energy, peak timing or latency, so it is the default forecaster.

Per building (test WAPE %): B3_Academic: Chronos-2 7.09, RF 7.89, prev-week 9.56, prev-day 13.20;
B5_Lab: Chronos-2 4.54, RF 4.95, prev-week 5.99, prev-day 6.73.

History threshold (validation, Chronos-2): 1 day 15.4%, 2 days 9.1%, 3 days 8.1%, 7 days 5.8%, 14 days 5.2% WAPE,
so routing requires 14 days (1,344 measured readings) before a building is forecast.

Fine-tuning was tried and dropped: a 300-step local LoRA fine-tune reached 5.44% test WAPE (5.24% validation), a gap
too small to matter on 60 forecasts, and Adaption Labs offers no Chronos training model for this account (live
read-only check, 2026-10-29). The fine-tuning code has been removed.

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

- Live tests (`tests/test_live_integration.py`) are opt-in and skipped in the default run.
- GPU inference was not tested.

## Reproduce

```bash
pip install -r requirements-forecast.txt
python -m src.generate_data && python -m src.pipeline
python -m src.forecasting.backtest
```
