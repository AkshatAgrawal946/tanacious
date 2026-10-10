<<<<<<< HEAD
# tanacious
=======
# Campus Energy – Prototype 1

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py          # writes data/raw/*.csv
jupyter notebook notebooks/01_baseline_eda.ipynb
```

- `src/generate_data.py` – synthetic hourly kWh for 5 buildings (Jun–Aug), with random multi-day gaps; true monthly totals saved separately.
- `notebooks/01_baseline_eda.ipynb` – resampling + 4–8 PM demand-spike chart.
>>>>>>> f9c9fcf (first commit)
