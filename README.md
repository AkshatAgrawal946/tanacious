# ⚡ Tanacious — Campus Energy Imputation Engine

> **AI Synergy Hackathon 2026** | **Case 05: Energy & Environment ("The 4 PM Spike")**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📖 Overview
**Tanacious** is a Data Integration & Load Imputation Engine designed to solve a fundamental energy blind spot for educational campuses. 

Campuses often experience a sharp electricity demand spike between 4 PM and 8 PM on weekdays, precisely when rooftop solar generation begins to drop. Facilities teams cannot safely shift or optimize this load due to a massive data gap: some buildings report granular 15-minute interval data, while others only report aggregate monthly utility bills. 

*You cannot optimize what you cannot measure.* Tanacious bridges this gap using machine learning to infer continuous 15-minute load curves for unmetered buildings, delivering a unified, real-time campus demand profile.

---

## 🎯 The Problem: "The Missing Meter"
* **Data Blind Spot:** Uneven meter granularity across campus buildings prevents accurate tracking of localized consumption[cite: 1].
* **Non-Linear Demand:** Electricity usage surges sharply in the late afternoon; a flat mathematical average of a monthly bill completely fails to capture the 4 PM – 8 PM spike.
* **Safety Constraints:** Facilities cannot automate load shedding without knowing exactly which buildings (e.g., flexible hostel cooling vs. critical research labs) are consuming power at specific times[cite: 2].

## 💡 The Solution Architecture
We divide our pipeline into three distinct tracks to securely simulate, impute, and visualize the campus grid[cite: 4].

### 1. The Smart Data Agent (Simulation Test Bed)
Because real campus data with this specific failure mode is restricted, we built a highly realistic synthetic data generator (`src/generate_data.py`)[cite: 4]. 
* Defines specific "Building Personas" (e.g., 24/7 Engineering Labs, or Hostels that spike at 7 AM and 6 PM)[cite: 4].
* Injects environmental variables like Temperature and Solar Radiation so cooling loads scale mathematically[cite: 4].
* Masks the 15-minute data for a subset of buildings to simulate the "Monthly Bill Only" constraint, saving the true data separately for final model evaluation[cite: 4].

### 2. The ML Imputation Pipeline (Random Forest Core)
* **Feature Engineering:** Extracts temporal context (Hour_of_Day, Day_of_Week, Is_Weekend) and combines it with environmental context (Temperature, Solar Radiation) and operational context (Building_Type)[cite: 4].
* **Tree-Based Inference:** Trains a Random Forest model exclusively on the metered buildings, then infers the 15-minute intervals for unmetered buildings to accurately capture non-linear demand spikes[cite: 7].
* **Smart Billing Reconciliation:** Applies a post-processing mathematical scaling rule to the AI's raw predictions to ensure the sum of the imputed 15-minute intervals perfectly matches the building's known monthly utility bill[cite: 7]. We utilize Additive Shifting or Peak Isolation to correct the baseline without artificially distorting the crucial 4 PM spike[cite: 7].

### 3. The Interactive Dashboard
* Built with Streamlit (`app.py`), the dashboard plots the combined load curve to provide decision-ready visibility to facilities managers[cite: 4].
* Isolates the 4 PM to 8 PM window to pinpoint which building types are driving the peak demand[cite: 4].
* Overlays the model's predicted load curve against the hidden ground truth data, verifying imputation accuracy using MAE and RMSE metrics[cite: 4].

---

## ⚙️ Tech Stack
* **Data Processing:** `pandas`, `numpy` (Time-series resampling, feature engineering)[cite: 2]
* **Machine Learning:** `scikit-learn` (Random Forest Regression)[cite: 2, 4]
* **Frontend/Dashboard:** `streamlit`[cite: 2, 4]

---

## 🗂 Project Structure
```text
tanacious/
├── 📁 data/                 # Generated datasets (data/raw/*.csv)
├── 📁 notebooks/
│   └── 01_baseline_eda.ipynb   # Resampling + 4–8 PM demand-spike chart
├── 📁 src/
│   └── generate_data.py        # Synthetic data generator / Smart Agent
├── 📄 app.py                   # Streamlit interactive dashboard
├── 📄 requirements.txt         # Project dependencies
├── 📄 Problem Statement Justifications.pdf
└── 📄 6abc0300be656_ai_synergy_hackathon_2026_round_1_case_file_redesigned.pdf
```

---

# Technical documentation

A campus electricity prototype that reconstructs 15-minute electricity profiles for buildings that only have monthly bills, evaluated on **synthetic** data. Every reconstructed building-month adds up exactly to its bill, and real meter readings are kept unchanged.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data                              # synthetic 2026 campus data -> data/raw/
python -m src.pipeline                                   # train, reconstruct, evaluate
python -m streamlit run app.py                           # dashboard (or .\run_app.ps1 on Windows)
pytest -q                                                # tests
```

`python -m src.ml_model` trains and saves only the Random Forest (`models/random_forest.joblib` plus its metadata sidecar, not committed). `app.py` is a Streamlit page: run it with `streamlit run`, not `python app.py`.

## Architecture

```
data/raw/*.csv ──► src/data.py        validation report: errors block, warnings surface
                      ▼
                   src/ml_model.py     Random Forest load shape
                   src/reconstruct.py  exact monthly bill reconciliation, real readings kept
                   src/evaluate.py     building-level holdout scenarios
                   src/pipeline.py     CLI: train, reconstruct, evaluate, write artifacts
                      ▼
                   src/provenance.py   dataset fingerprints, environment, JSON artifacts
                      ▼
                   src/simulate.py     what-if load shifting and trimming, tariff
                   app.py              Streamlit: Campus overview · Building · Simulate
```

## Data contracts

`interval_15min.csv`: `timestamp, building, building_type, temperature, kWh_consumed`
`monthly_totals.csv`: `building, building_type, month (YYYY-MM), monthly_kWh`

Rules (`src/data.py`):

- **Errors** (`load_data` raises `DataValidationError` with examples): missing columns; unparseable, timezone-aware or off-grid (not 15-minute aligned) timestamps; duplicate building/timestamp rows; negative, infinite or non-numeric energy; blank building ids; malformed months; duplicate or blank bills; a building with more than one type, or different types in the two files; metered buildings with no bills; a month whose metered energy exceeds its bill by more than `BILL_TOLERANCE_KWH` (0.01 kWh); a fully metered month that does not sum to its bill.
- **Warnings** (allowed): blank readings and absent rows inside a metered span (sensor dropouts, pilot meters), missing temperatures, missing bill months, metered months with no bill.
- Timestamps are timezone-naive campus local time (no DST shifts).
- `ground_truth_15min.csv` is synthetic evaluation truth. It is only read by `src.evaluate.load_ground_truth`, which tags the frame so training refuses it.

The generator (`src/generate_data.py`) creates all of 2026 (12 bills per building) with seasonal temperature. Academic calendar: `BREAKS` in `src/campus_calendar.py` (summer vacation 16 May - 19 July, winter break 19 Dec - 3 Jan; edit to match the college). Pilot meters (`PILOT_METERS`): Hostel A and the Library get a temporary meter for 8-21 June and 3-16 August.

## Reconstruction

The Random Forest learns how load shape follows each building type's schedule (`TYPE_SCHEDULES`, `TYPE_CALENDAR`: coarse facilities estimates to edit per campus), weekends, temperature (measured, interpolated) and academic breaks. Clock time is deliberately not a feature. Each building's vacation level is read from its own bills (`break_ratios`). The bill then fixes the absolute level:

- reconstructed + observed energy equals the monthly bill (within 0.01 kWh);
- observed readings are returned bit-for-bit unchanged and only missing intervals are estimated;
- no negative, NaN or infinite values; zero-kWh months work;
- duplicate readings, off-grid readings, duplicate bills, shape-length mismatches and over-bill meters fail with explicit errors.

These invariants are covered by parametrised tests over 28/29/30/31-day months and 0-100% metering (`tests/test_reconstruct_invariants.py`).

Information note: `break_ratios()` and the RF target normalisation use all of a building's bills, including later months. That is legitimate for reconstructing completed months, but this model must not be used to predict the future.

## Evaluation

`python -m src.pipeline` (no hyperparameters are tuned on any scenario):

- *Bill-only buildings*: scored against the hidden synthetic truth outside the pilot-meter weeks; pilot readings must come back unchanged (checked, `observed_max_abs_change_kwh` = 0).
- *Unmetered holdout*: each metered building is rebuilt from the other buildings plus its own bill.
- *Partially metered*: a metered building keeps 0/25/50/75% of its days; the rest are reconstructed and scored. The RF is compared with a flat bill split.

Metrics (`src/metrics.py`): MAE, RMSE, WAPE (headline), MAPE (denominator floored at 0.01 kWh), 4-8 PM energy error, interval-level 4-8 PM MAE.

Outputs: `data/processed/evaluation_metrics.csv`, `evaluation_15min.csv`, `reconstruction_partial_metering.csv`, and `evaluation_metrics.meta.json` (dataset fingerprint, model version, environment).

## Dashboard

`python -m streamlit run app.py`. Tabs: **Campus overview** (all buildings stacked, monthly energy, summary table), **Building** (one line per selected month, 15-minute load, monthly energy, tested accuracy) and **Simulate** (what-if load shifting).

**Simulate** (`src/simulate.py`) splits each building's load into a flat base load (5th percentile of 15-minute demand, never touched) and the flexible HVAC load above it, then applies a scenario to the selected months:

- *Trim*: removes a share of flexible load at every hour (a setpoint raise, better controls).
- *Move*: takes a share of the flexible 4-8 PM load and pre-cools it into an earlier window of the same day, plus a loss on the energy moved. The moved load fills the quietest intervals of the **campus** total first, so buildings do not pile into the same hours; each building-day keeps its own energy apart from the loss and the trim.
- *Defaults*: move 25% into 10:00-16:00, 10% loss, no trim (`simulate.Scenario`). On the synthetic year this is the largest share that still gives the full cut in campus peak (-4.6% in an average month, -2.9% for the year) with no month's peak rising; beyond it the campus peak sits after 8 PM, so moving more only adds losses. The search covered windows ending at 4 PM and starting at most 6 hours earlier: earlier windows score better only because the flat loss ignores that a building cannot hold cooling that long.
- *Cost*: time-of-use energy rates plus a demand charge on each building's monthly maximum 15-minute load. The default rates (`simulate.Tariff`, currency `CURRENCY` in `app.py`) are placeholders.
- *Colours*: every change is red when it is a loss (more peak, energy or cost), grey when nothing changes and green when it is a gain, in the metrics and the building table.

It is a planning sketch on reconstructed load, not a controls model: it assumes buildings can store the moved cooling, and for bill-only buildings the flexible load is itself an estimate.

Caches are keyed by content hashes of the input files and dataset plus the model version; a saved Random Forest is reused only if its metadata sidecar (`models/random_forest.meta.json`) matches the dataset fingerprint, model version, features, calendar and scikit-learn version. Accuracy badges are shown only if the stored evaluation was computed on the current data.

## Results on the synthetic data

See [RESULTS.md](RESULTS.md).

## Limitations

- All data is synthetic: smooth schedules, one shared temperature series, Gaussian noise, a fixed calendar. Scores say nothing about real-campus accuracy.
- The model's schedules and calendar factors were designed with knowledge of the generator, and its design was iterated while looking at these same evaluation scores; its evaluation is therefore optimistic.
- Only two buildings are continuously metered, so the unmetered and partially metered tests rest on two buildings.

## Before a real-campus deployment

Real interval and billing data with their actual quirks (meter replacements, estimated bills, billing periods that do not follow calendar months, DST); re-running the evaluation on that data; access control for the dashboard.
