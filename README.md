# The Missing Meter - MVP

A hackathon-ready campus energy prototype that reconstructs 15-minute electricity profiles for buildings with only monthly bills. Every reconstructed building-month is reconciled to the known bill exactly.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data   # synthetic campus data
python -m src.pipeline        # reconstruct, train + save the Random Forest, compare both models
.\run_app.ps1
```

`python -m src.ml_model` trains and saves only the Random Forest (`models/random_forest.joblib`, not committed).

Run tests with `pytest -q`.

`app.py` is a Streamlit page, not a standalone Python script. Running `python app.py` starts it in Streamlit's "bare mode" and only produces warnings; use `run_app.ps1` (or `python -m streamlit run app.py`) to start the local web server and open the dashboard.

## What is implemented

- CSV schema and data-quality validation
- 15-minute profile baseline: same-type smart-meter profile when one exists, otherwise the metered profile re-timed by a building-type daily schedule (academic peaks 2-4 PM, library and hostels in the evening)
- Monthly-bill reconciliation (`sum(interval kWh) == monthly_kWh`)
- Observed versus reconstructed labels
- A masked-meter evaluation harness with MAE, MAPE, peak-window error and bill-total error
- Random Forest alternative (`src/ml_model.py`): learns how load follows each building type's schedule and the temperature from the metered buildings, then predicts the shape for bill-only buildings. Clock time is deliberately not an input: the metered buildings (academic, lab) both peak in the afternoon, and with clock time the forest copies that peak onto hostels and the library
- Model comparison (`src/compare.py`): both models through the same bill reconciliation, scored against hidden truth for bill-only buildings and by holding out each metered building
- Streamlit dashboard that reads the campus data directly (generating it on first launch). Pick a building, up to 8 months, weekdays/weekends/all days and a model; charts show average power in kW with y-axes fixed to the busiest month of the year, so selections can be compared by eye. Tabs: "Campus overview" (all buildings stacked, monthly energy by building, summary table), "Building" (one line per selected month, 15-minute load, monthly energy) and "Compare models" (scores and charts for the selected months)

## Input schema

`interval_15min.csv`: `timestamp`, `building`, `building_type`, `temperature`, `kWh_consumed`.

`monthly_totals.csv`: `building`, `building_type`, `month`, `monthly_kWh`.

The bundled generator creates `data/raw/` files for all of 2026 (12 monthly bills per building), with seasonal temperature (hottest late May, coolest in winter). `ground_truth_15min.csv` is demo-only and must never be used by the reconstruction pipeline.

Academic calendar (`BREAKS` in `src/campus_calendar.py`, currently summer vacation 16 May - 19 July and winter break 19 Dec - 3 Jan; edit to match the college): occupancy drops during breaks, and so does the always-on load (`BREAK_STANDBY`: closed blocks switch most lighting, servers and AC off; labs keep running). Both models are told which days are breaks.

Pilot meters (`PILOT_METERS` in `src/generate_data.py`): Hostel A and the Library get a temporary clamp-on meter for two fortnights, 8-21 June (summer vacation) and 3-16 August (term time), so `interval_15min.csv` holds four weeks of their 15-minute readings. Both models use these weeks (they are the only data showing what an evening-peaking building looks like, busy and near-empty); the comparison scores both models only outside them.

## Demo claim

This is synthetic data. The reconstructed curve is an operational estimate, not a measurement. The app makes the bill-total constraint explicit and provides a masked-meter evaluation route before any real-world claim is made.
