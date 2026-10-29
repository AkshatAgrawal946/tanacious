# Reconstruction results (synthetic data, 2026-10-29)

Every number below was measured on the synthetic 2026 dataset on a CPU-only Mac (Python 3.14.7, scikit-learn 1.9.1).
They are not real-campus accuracy. Provenance is in `data/processed/evaluation_metrics.meta.json`.

## Building-level holdouts

`python -m src.pipeline`; WAPE / MAPE % on reconstructed intervals:

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

## Reproduce

```bash
python -m src.generate_data && python -m src.pipeline
```
