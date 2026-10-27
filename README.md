# The Missing Meter - MVP

A hackathon-ready campus energy prototype that reconstructs 15-minute electricity profiles for buildings with only monthly bills. Every reconstructed building-month is reconciled to the known bill exactly.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data
python -m src.pipeline
.\run_app.ps1
```

Run tests with `pytest -q`.

`app.py` is a Streamlit page, not a standalone Python script. Running `python app.py` starts it in Streamlit's "bare mode" and only produces warnings; use `run_app.ps1` (or `python -m streamlit run app.py`) to start the local web server and open the dashboard.

## What is implemented

- CSV schema and data-quality validation
- 15-minute profile baseline: same-type smart-meter profile when one exists, otherwise the metered profile re-timed by a building-type daily schedule (academic peaks 2-4 PM, library and hostels in the evening)
- Monthly-bill reconciliation (`sum(interval kWh) == monthly_kWh`)
- Observed versus reconstructed labels
- A masked-meter evaluation harness with MAE, MAPE, peak-window error and bill-total error
- Streamlit dashboard that reads the campus data directly (generating it on first launch), with building selection, a peak-window chart and quality checks

## Input schema

`interval_15min.csv`: `timestamp`, `building`, `building_type`, `temperature`, `kWh_consumed`.

`monthly_totals.csv`: `building`, `building_type`, `month`, `monthly_kWh`.

The bundled generator creates `data/raw/` files. `ground_truth_15min.csv` is demo-only and must never be used by the reconstruction pipeline.

## Demo claim

This is synthetic data. The reconstructed curve is an operational estimate, not a measurement. The app makes the bill-total constraint explicit and provides a masked-meter evaluation route before any real-world claim is made.
