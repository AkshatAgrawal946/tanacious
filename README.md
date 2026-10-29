# The Missing Meter (Tanacious)

A campus electricity prototype with two separate machine-learning tasks, both evaluated on **synthetic** data:

| | Task A: reconstruction | Task B: forecasting |
|---|---|---|
| Question | What did a building use, every 15 minutes, in a past month? | What will a metered building use over the next 24 hours? |
| Inputs | Monthly bills, building type, academic calendar, weather, other buildings' meters, any partial readings | The building's own measured 15-minute history (14 days), calendar covariates known in advance |
| Model | Random Forest load shape + exact monthly bill reconciliation (`src/ml_model.py`, `src/reconstruct.py`) | Amazon Chronos-2 (zero-shot) vs. seasonal and Random Forest baselines (`src/forecasting/`) |
| Evaluation | Building-level holdouts: unmetered and partially metered (`src/evaluate.py`) | Chronological rolling origins with train / validation / test months (`src/forecasting/backtest.py`) |

A building with no meter history is never "forecast": an autoregressive model has nothing to condition on, so bill-only periods are always reconstructed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                          # reconstruction + forecasting baselines
pip install -r requirements-forecast.txt                 # optional: Chronos-2 (torch, chronos-forecasting)
python -m src.generate_data                              # synthetic 2026 campus data -> data/raw/
python -m src.pipeline                                   # Task A: train, reconstruct, evaluate
python -m src.forecasting.backtest                       # Task B: benchmark (add --skip-chronos for baselines only)
python -m streamlit run app.py                           # dashboard (or .\run_app.ps1 on Windows)
pytest -q                                                # tests (live provider tests are opt-in, see below)
```

The first Chronos-2 run downloads the public `amazon/chronos-2` weights (~480 MB) from Hugging Face (set `HF_TOKEN` in the shell for faster downloads). Inference runs on CPU by default (`device` in `configs/forecast_benchmark.json`; `"auto"` picks CUDA when present).

## Architecture

```
data/raw/*.csv ──► src/data.py (validation report: errors block, warnings surface)
                      │
       ┌──────────────┴──────────────────────┐
       ▼                                     ▼
 Task A: reconstruction                Task B: forecasting (src/forecasting/)
 ml_model.py  RF load shape            data_prep.py   leak-free ForecastRequests
 reconstruct.py  bill reconciliation   baselines.py   seasonal day/week, chronological RF
 evaluate.py  holdout scenarios        chronos_adapter.py  zero-shot Chronos-2 (lazy load, timeout)
 pipeline.py  CLI                      backtest.py    rolling-origin benchmark + promotion gate
                                       selection.py   routing + explicit fallback
       └──────────────┬──────────────────────┘
                      ▼
 src/provenance.py  dataset fingerprints, environment, JSON artifacts
                      ▼
 app.py  Streamlit: Campus overview · Building · Forecast
```

## Data contracts

`interval_15min.csv`: `timestamp, building, building_type, temperature, kWh_consumed`
`monthly_totals.csv`: `building, building_type, month (YYYY-MM), monthly_kWh`

Rules (`src/data.py`):

- **Errors** (`load_data` raises `DataValidationError` with examples): missing columns; unparseable, timezone-aware or off-grid (not 15-minute aligned) timestamps; duplicate building/timestamp rows; negative, infinite or non-numeric energy; blank building ids; malformed months; duplicate or blank bills; a building with more than one type, or different types in the two files; metered buildings with no bills; a month whose metered energy exceeds its bill by more than `BILL_TOLERANCE_KWH` (0.01 kWh); a fully metered month that does not sum to its bill.
- **Warnings** (allowed): blank readings and absent rows inside a metered span (sensor dropouts, pilot meters), missing temperatures, missing bill months, metered months with no bill.
- Timestamps are timezone-naive campus local time (no DST shifts).
- `ground_truth_15min.csv` is synthetic evaluation truth. It is only read by `src.evaluate.load_ground_truth`, which tags the frame so training refuses it.

## Task A: reconstruction

The Random Forest learns how load shape follows each building type's schedule (`TYPE_SCHEDULES`, `TYPE_CALENDAR`: coarse facilities estimates to edit per campus), weekends, temperature (measured, interpolated) and academic breaks (`src/campus_calendar.py`). Clock time is deliberately not a feature. Each building's vacation level is read from its own bills (`break_ratios`). The bill then fixes the absolute level:

- reconstructed + observed energy equals the monthly bill (within 0.01 kWh);
- observed readings are returned bit-for-bit unchanged and only missing intervals are estimated;
- no negative, NaN or infinite values; zero-kWh months work;
- duplicate readings, off-grid readings, duplicate bills, shape-length mismatches and over-bill meters fail with explicit errors.

These invariants are covered by parametrised tests over 28/29/30/31-day months and 0-100% metering (`tests/test_reconstruct_invariants.py`).

Information note: `break_ratios()` and the RF target normalisation use all of a building's bills, including later months. That is legitimate for retrospective reconstruction of completed months, but not at a forecast origin, so this model is never used for forecasting.

Evaluation (`python -m src.pipeline`):

- *Bill-only buildings*: scored against the hidden synthetic truth outside the pilot-meter weeks; pilot readings must come back unchanged (checked, `observed_max_abs_change_kwh` = 0).
- *Unmetered holdout*: each metered building is rebuilt from the other buildings plus its own bill.
- *Partially metered*: a metered building keeps 0/25/50/75% of its days; the rest are reconstructed and scored. The RF is compared with a flat bill split.

Outputs: `data/processed/evaluation_metrics.csv`, `evaluation_15min.csv`, `reconstruction_partial_metering.csv`, and `evaluation_metrics.meta.json` (dataset fingerprint, model version, environment).

## Task B: forecasting

Protocol (`configs/forecast_benchmark.json`, versioned and validated by `ForecastExperimentConfig`):

- buildings `B3_Academic`, `B5_Lab` (the continuously metered ones); 15-minute steps; 1,344-step (14-day) context; 96-step (24 h) horizon; one origin per day at midnight;
- splits: train Jan-Jul 2026, validation Aug 2026, test (final holdout) Sep 2026. These are experimental choices for the synthetic year, not production defaults;
- every model gets the identical `ForecastRequest` objects and is scored on the identical measured intervals (asserted); origins with >50% missing targets are skipped for all models;
- covariates: time of day, weekday, weekend, academic break (all known in advance). Measured future temperature is **not** used; the `calendar+oracle_temperature` scenario exists only as an explicitly declared perfect-weather-forecast experiment. Monthly bills are never used for forecasting;
- `ForecastRequest` refuses history that reaches the origin and future covariates without past values; `validate_result` rejects misaligned timestamps, wrong shapes, NaN and crossing quantiles.

Models:

| Name | What it is |
|---|---|
| `seasonal_previous_day` | same time yesterday (falls back further back over dropouts) |
| `seasonal_previous_week` | same weekday and time last week |
| `random_forest_forecast` | chronological direct multi-step RF on lagged history + calendar, trained on training-split origins only |
| `chronos2_zero_shot` | `amazon/chronos-2` with calendar covariates, P10/P50/P90 |

Metrics (`src/metrics.py`): MAE, RMSE, WAPE (headline), MAPE (denominator floored at 0.01 kWh), 4-8 PM energy error, interval-level 4-8 PM MAE, daily peak magnitude and timing error, P10-P90 coverage, per-forecast latency and process peak memory.

Promotion gate (configurable engineering rule, not a performance claim): the candidate must beat the strongest simple baseline (chosen on **validation**) by at least 10% relative WAPE on the **test** month, without worsening 4-8 PM energy error by more than 1 percentage point or peak timing by more than 30 minutes, at no more than 5 s per forecast. Otherwise the baseline stays the default.

History threshold for routing: on the validation month, the default model is run with 1, 2, 3, 7 and 14 days of history; the threshold is the shortest history whose WAPE is within 5% of the 14-day WAPE. `selection.route` forecasts a building only when it has that many *measured* readings before the origin; otherwise it is reconstructed. Without a benchmark, a conservative 14-day default is used and logged.

Artifacts (`data/processed/forecast_benchmark/`): `interval_predictions.csv`, `scores.csv`, `summary.json` (config and hash, dataset fingerprint, environment, every run's status including `not_run` reasons, gate decisions, default model, history threshold).

### Why Chronos-2 is not fine-tuned

Zero-shot Chronos-2 already passes the promotion gate by a wide margin, and a 300-step local LoRA fine-tune (tried on
2026-10-29) improved test WAPE only from 5.50% to 5.44%, which is within noise on 60 forecasts. Adaption Labs, the
intended hosted fine-tuning provider, does not offer a Chronos training model for this account (live check on the same
day). Fine-tuning code was therefore removed; `Chronos2Pipeline.fit` in `chronos-forecasting` is the route if it is ever
needed on real data.

## Dashboard

`python -m streamlit run app.py`. Tabs: **Campus overview** and **Building** (reconstructed months, 4-8 PM share, tested accuracy) and **Forecast** (pick a metered building, a day and a model, or Automatic = the benchmark's default). The forecast chart shows real meter readings (solid blue), reconstructed history (dotted grey) and the forecast (orange, with P10-P90 band), plus the model name and version, its benchmark WAPE and coverage. If the chosen model cannot run, the previous-week baseline is used and a warning names the reason. Buildings without enough measured history are routed to reconstruction with an explanation. The app never starts training.

Caches are keyed by content hashes of the input files and dataset plus the model version; a saved Random Forest is reused only if its metadata sidecar (`models/random_forest.meta.json`) matches the dataset fingerprint, model version, features, calendar and scikit-learn version. Accuracy badges are shown only if the stored evaluation was computed on the current data.

## Tests

`pytest -q` runs everything offline (Chronos-2 is mocked). Live tests are opt-in and skipped otherwise; a skip means *unverified*, not passed:

```bash
TANACIOUS_LIVE_CHRONOS=1 pytest tests/test_live_integration.py
```

## Results on the synthetic data

See [RESULTS.md](RESULTS.md) for the measured numbers and the commands that produced them.

## Limitations

- All data is synthetic (`src/generate_data.py`): smooth schedules, one shared temperature series, Gaussian noise, a fixed calendar. Scores say nothing about real-campus accuracy, and model rankings may not transfer.
- The reconstruction model's schedules and calendar factors were designed with knowledge of the generator, and its design was iterated while looking at these same evaluation scores; its evaluation is therefore optimistic.
- Only two buildings are continuously metered, so the forecasting benchmark has 2 series and ~60 test forecasts; differences of a few tenths of a WAPE point are not meaningful.
- No weather forecasts are available, so forecasts use calendar covariates only.
- Forecast origins are daily at midnight; intra-day re-forecasting is not evaluated.

## Before a real-campus deployment

Real interval and billing data with their actual quirks (meter replacements, estimated bills, billing periods that do not follow calendar months, DST); a weather-forecast feed; re-running both evaluations and the promotion gate on that data (and re-testing whether fine-tuning helps there); monitoring of forecast error and fallbacks; access control for the dashboard.
