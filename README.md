# The Missing Meter - MVP

A hackathon-ready campus energy prototype that reconstructs 15-minute electricity profiles for buildings with only monthly bills. Every reconstructed building-month is reconciled to the known bill exactly.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data   # synthetic campus data
python -m src.pipeline        # train + save the Random Forest, reconstruct, check accuracy
.\run_app.ps1
```

`python -m src.ml_model` trains and saves only the Random Forest (`models/random_forest.joblib`, not committed).

Run tests with `pytest -q`.

`app.py` is a Streamlit page, not a standalone Python script. Running `python app.py` starts it in Streamlit's "bare mode" and only produces warnings; use `run_app.ps1` (or `python -m streamlit run app.py`) to start the local web server and open the dashboard.

## What is implemented

- CSV schema and data-quality validation
- Random Forest model (`src/ml_model.py`): learns from every real 15-minute reading how load follows each building type's daily schedule, weekend activity, the temperature and the academic calendar, then predicts the load shape for every building. Building-type knowledge (`TYPE_SCHEDULES`, `TYPE_CALENDAR`) is a coarse facilities estimate to edit for the college. Clock time is deliberately not an input: the forest would copy the metered buildings' afternoon peak onto hostels and the library
- Vacation level from the bills (`break_ratios`): each building's break-day energy relative to a term day is read from its own all-break versus all-term monthly bills and applied on break days, so months that are part term and part vacation (May, July, December) are split correctly even for building types the forest has not seen metered
- Monthly-bill reconciliation (`sum(interval kWh) == monthly_kWh`); real meter readings are kept and only the gaps are estimated
- Observed versus reconstructed labels
- Accuracy check (`src/evaluate.py`): bill-only buildings are scored against the hidden truth (pilot weeks excluded) and each metered building is rebuilt with its own meter hidden. Results go to `data/processed/evaluation_metrics.csv`
- Streamlit dashboard that reads the campus data directly (generating it on first launch). Pick a building, up to 8 months and weekdays/weekends/all days; charts show average power in kW with y-axes fixed to the busiest month of the year, so selections can be compared by eye. Tabs: "Campus overview" (all buildings stacked, monthly energy by building, summary table) and "Building" (one line per selected month, 15-minute load, monthly energy, tested accuracy)

## Input schema

`interval_15min.csv`: `timestamp`, `building`, `building_type`, `temperature`, `kWh_consumed`.

`monthly_totals.csv`: `building`, `building_type`, `month`, `monthly_kWh`.

The bundled generator creates `data/raw/` files for all of 2026 (12 monthly bills per building), with seasonal temperature (hottest late May, coolest in winter). `ground_truth_15min.csv` is demo-only and must never be used by the reconstruction pipeline.

Academic calendar (`BREAKS` in `src/campus_calendar.py`, currently summer vacation 16 May - 19 July and winter break 19 Dec - 3 Jan; edit to match the college): occupancy drops during breaks, and so does the always-on load (`BREAK_STANDBY`: closed blocks switch most lighting, servers and AC off; labs keep running). The model is told which days are breaks.

Pilot meters (`PILOT_METERS` in `src/generate_data.py`): Hostel A and the Library get a temporary clamp-on meter for two fortnights, 8-21 June (summer vacation) and 3-16 August (term time), so `interval_15min.csv` holds four weeks of their 15-minute readings. The model trains on these weeks (they are the only data showing what an evening-peaking building looks like, busy and near-empty); the accuracy check scores it only outside them.

## Demo claim

This is synthetic data. The reconstructed curve is an operational estimate, not a measurement. The app makes the bill-total constraint explicit and the accuracy check (hidden-meter and hidden-truth scoring) should be repeated on real data before any real-world claim is made.
