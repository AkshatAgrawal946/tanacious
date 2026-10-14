# Build chat log – Campus Energy Prototype 1

> Step 1 (Scaffolding): Prompt Claude Code to set up a standard data science directory: Create a Python project structure with folders for data, notebooks, src, and a requirements.txt including pandas, scikit-learn, and streamlit.
> 
> Step 2 (Synthetic Data Generation): Because you lack actual campus data, prompt Claude Code to build a data generator: Write a Python script using pandas to generate a synthetic dataset of hourly electricity consumption for 5 buildings over 3 months. Include columns for timestamp, temperature, building_type, and kWh_consumed. Add random missing data chunks to simulate the 'monthly total' constraint.
> 
> Step 3 (Baseline Notebook): Have Claude Code draft a Jupyter Notebook template: Generate a Jupyter Notebook that loads the synthetic dataset, performs basic time-series resampling, and outputs a Matplotlib line chart showing the 4 PM to 8 PM demand spike.we are just making the first  prototype so dont go very overboard.

⏺ I'll scaffold the project, write a small data generator, and build a baseline notebook, keeping all of it minimal. First I'll check what's already in the directory.

⏺ **Bash**(List directory and check Python packages)
```bash
ls -la && python3 --version && python3 -c "import pandas, numpy, matplotlib, nbformat; print('ok')"
```

⏺ The folder only has the two case-file PDFs, it isn't a git repo, and the system Python has no pandas. I'll write the files in place, then test everything in a local `.venv`.

⏺ **Bash**(Create folders, requirements, gitignore)
```bash
mkdir -p data/raw data/processed notebooks src && touch data/raw/.gitkeep data/processed/.gitkeep src/__init__.py && cat > requirements.txt <<'EOF'
pandas
numpy
scikit-learn
streamlit
matplotlib
jupyter
EOF
cat > .gitignore <<'EOF'
.venv/
__pycache__/
.ipynb_checkpoints/
data/raw/*.csv
data/processed/*.csv
EOF
```

⏺ **Write**(/Users/akshatagrawal/Projects/tanacious/prot1/src/generate_data.py)
```python
"""Generate a synthetic hourly electricity dataset for 5 campus buildings over 3 months.

Outputs:
  data/raw/hourly_consumption.csv  - hourly readings, with random multi-day gaps (NaN kWh)
  data/raw/monthly_totals.csv      - true monthly kWh per building (what the meter bill shows)

The gaps simulate buildings where only the monthly total is reliably known.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2026-06-01", "2026-08-31 23:00"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (building_type, base load kWh, evening peak extra kWh)
BUILDINGS = {
    "B1_Hostel_A": ("hostel", 40, 60),
    "B2_Hostel_B": ("hostel", 35, 55),
    "B3_Academic": ("academic", 30, 25),
    "B4_Library": ("library", 25, 30),
    "B5_Lab": ("lab", 50, 20),
}


def make_temperature(ts: pd.DatetimeIndex, rng) -> np.ndarray:
    """Summer temperature: daily cycle peaking ~3 PM, plus noise."""
    daily = 6 * np.sin(2 * np.pi * (ts.hour - 9) / 24)
    return 30 + daily + rng.normal(0, 1.5, len(ts))


def make_load(ts, temp, base, peak, rng) -> np.ndarray:
    hour = ts.hour.to_numpy()
    # 4 PM - 8 PM spike (smooth bump centred at 6 PM)
    evening = peak * np.exp(-((hour - 18) ** 2) / 4)
    daytime = 0.4 * peak * ((hour >= 9) & (hour <= 17))
    cooling = 1.5 * np.clip(temp - 26, 0, None)  # AC load above 26 C
    weekend = np.where(ts.dayofweek >= 5, 0.8, 1.0)
    load = (base + evening + daytime + cooling) * weekend
    return np.clip(load + rng.normal(0, 3, len(ts)), 0, None)


def add_missing_chunks(kwh: np.ndarray, rng, n_chunks=3, min_days=2, max_days=7) -> np.ndarray:
    kwh = kwh.copy()
    for _ in range(n_chunks):
        length = rng.integers(min_days, max_days + 1) * 24
        start = rng.integers(0, len(kwh) - length)
        kwh[start : start + length] = np.nan
    return kwh


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, END, freq="h")
    temp = make_temperature(ts, rng)

    frames = []
    for name, (btype, base, peak) in BUILDINGS.items():
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "building": name,
            "building_type": btype,
            "temperature": temp.round(1),
            "kWh_consumed": make_load(ts, temp, base, peak, rng).round(2),
        }))
    full = pd.concat(frames, ignore_index=True)

    # Monthly totals come from the complete data (the "bill" is always known)
    monthly = (full.groupby(["building", full["timestamp"].dt.to_period("M")])["kWh_consumed"]
               .sum().round(1).reset_index().rename(columns={"timestamp": "month", "kWh_consumed": "monthly_kWh"}))

    # Knock out random multi-day chunks of hourly data per building
    full["kWh_consumed"] = full.groupby("building")["kWh_consumed"].transform(
        lambda s: add_missing_chunks(s.to_numpy(), rng))
    return full, monthly


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly, monthly = generate()
    hourly.to_csv(OUT_DIR / "hourly_consumption.csv", index=False)
    monthly.to_csv(OUT_DIR / "monthly_totals.csv", index=False)
    print(f"hourly: {len(hourly):,} rows, {hourly['kWh_consumed'].isna().mean():.1%} missing kWh")
    print(f"monthly: {len(monthly)} rows -> {OUT_DIR}")

```

⏺ Next is the notebook. I'll build it with a small nbformat script so the JSON is valid.

⏺ **Bash**(Create venv and install requirements)
```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt 2>&1 | tail -3; .venv/bin/python -c "import pandas, sklearn, streamlit, matplotlib, nbformat; print('deps ok')"
```

⏺ **Bash**(Build notebook, generate data, execute notebook)
```bash
cat > /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py <<'PYEOF'
import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("# 01 · Baseline exploration\nLoads the synthetic campus dataset, resamples it, and plots the **4 PM – 8 PM evening demand spike**.\n\nRun `python src/generate_data.py` first if `data/raw/` is empty."),
code("""import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw/hourly_consumption.csv", parse_dates=["timestamp"])
monthly = pd.read_csv("../data/raw/monthly_totals.csv")
df.head()"""),
md("## Missing data\nHourly gaps simulate buildings where only the monthly bill is known."),
code("""df.groupby("building")["kWh_consumed"].apply(lambda s: s.isna().mean()).rename("missing_frac").round(3)"""),
md("## Resampling\nHourly → daily totals (per building) and average hourly profile."),
code("""wide = df.pivot(index="timestamp", columns="building", values="kWh_consumed")

daily = wide.resample("D").sum(min_count=1)       # NaN for days fully missing
weekly = wide.resample("W").mean()

ax = daily.plot(figsize=(12, 4), title="Daily kWh per building")
ax.set_ylabel("kWh / day")
plt.show()"""),
md("## The 4 PM – 8 PM demand spike\nAverage load by hour of day, campus-wide and per building."),
code("""profile = df.groupby([df["timestamp"].dt.hour, "building"])["kWh_consumed"].mean().unstack()
campus = profile.sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(16, 20, color="tab:red", alpha=0.12, label="Peak window 4–8 PM")
ax.plot(campus.index, campus.values, color="black", lw=2.5, label="Campus total")
for b in profile.columns:
    ax.plot(profile.index, profile[b], lw=1, alpha=0.7, label=b)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Avg kWh")
ax.set_title("Average hourly demand profile")
ax.legend(fontsize=8, ncol=2)
plt.show()

peak_share = campus.loc[16:20].sum() / campus.sum()
print(f"4–8 PM accounts for {peak_share:.1%} of daily campus consumption")"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/01_baseline_eda.ipynb")
PYEOF
.venv/bin/python /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py && .venv/bin/python src/generate_data.py && cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 01_baseline_eda.ipynb --output /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed.ipynb 2>&1 | tail -2 && grep -o '4–8 PM accounts[^"\\]*' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed.ipynb
```

⏺ **Bash**(Fix peak window, rerun notebook, add README)
```bash
cd /Users/akshatagrawal/Projects/tanacious/prot1 && sed -i '' 's/campus.loc\[16:20\]/campus.loc[16:19]  # hours starting 4,5,6,7 PM/' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py && .venv/bin/python /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py && cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 01_baseline_eda.ipynb --output /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed.ipynb 2>/dev/null; grep -o '4–8 PM accounts for [0-9.]*%' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed.ipynb; cd .. && cat > README.md <<'EOF'
# Campus Energy – Prototype 1

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py          # writes data/raw/*.csv
jupyter notebook notebooks/01_baseline_eda.ipynb
```

- `src/generate_data.py` – synthetic hourly kWh for 5 buildings (Jun–Aug), with random multi-day gaps; true monthly totals saved separately.
- `notebooks/01_baseline_eda.ipynb` – resampling + 4–8 PM demand-spike chart.
EOF
find . -path ./.venv -prune -o -type f -print | sort
```

⏺ The peak share still reads 30.5%, which is suspicious after narrowing the window. The rerun may have failed silently, so I'll check.

⏺ **Bash**(Re-execute notebook and check output)
```bash
cd /Users/akshatagrawal/Projects/tanacious/prot1/notebooks && grep -c "16:19" 01_baseline_eda.ipynb; ../.venv/bin/jupyter nbconvert --to notebook --execute 01_baseline_eda.ipynb --output /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed2.ipynb 2>&1 | grep -v WARNING; grep -o '4–8 PM accounts for [0-9.]*%' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed2.ipynb
```

⏺ My sed edit put the comment in the middle of the expression and broke it. I'll repair the line and rerun the notebook.

⏺ **Bash**(Fix the line and re-execute notebook)
```bash
sed -i '' 's|peak_share = campus.loc\[16:19\]  # hours starting 4,5,6,7 PM.sum() / campus.sum()|peak_share = campus.loc[16:19].sum() / campus.sum()  # hours starting 4-7 PM|' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py && .venv/bin/python /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/make_nb.py && cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 01_baseline_eda.ipynb --output /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed3.ipynb 2>&1 | grep -E "Error|Writing"; grep -o '4–8 PM accounts for [0-9.]*%' /Users/akshatagrawal/.claude/jobs/e03332dd/tmp/executed3.ipynb
```

⏺ I've set up the prototype project, written the data generator and built the baseline notebook. I ran the generator and the notebook end to end in a local `.venv`, and both finished without errors.

**1. Project layout** (in `/Users/akshatagrawal/Projects/tanacious/prot1`, next to your case-file PDFs)
- Folders: `data/raw/`, `data/processed/`, `notebooks/`, `src/`.
- `requirements.txt` lists pandas, numpy, scikit-learn, streamlit, matplotlib and jupyter. I added matplotlib for the chart and jupyter to run the notebook.
- A short `README.md` with the run commands, and a `.gitignore`.

**2. Data generator** (`src/generate_data.py`)
- Produces hourly data for 5 buildings (2 hostels, an academic block, a library and a lab) from June to August 2026, which is 11,040 rows.
- Columns: `timestamp`, `building`, `building_type`, `temperature`, `kWh_consumed`.
- Load is built from a base level, a 4–8 PM evening bump, daytime use, extra cooling load when it's hot, and lower use at weekends.
- Each building has three random gaps of 2–7 days with no readings, so about 14% of kWh values are missing.
- The true monthly totals are saved separately in `data/raw/monthly_totals.csv`. This is the "monthly total" constraint: hourly detail is patchy, but the monthly bill is always known.

**3. Notebook** (`notebooks/01_baseline_eda.ipynb`)
- Loads the data and shows how much is missing for each building.
- Resamples hourly data to daily totals and plots them.
- Draws a Matplotlib chart of average demand by hour of day, campus total and per building, with the 4–8 PM window shaded.
- On the generated data, 4–8 PM accounts for **26.0%** of daily campus consumption.

result: Prototype project created in `prot1/`: folder structure and `requirements.txt`, a generator for 3 months of hourly data for 5 buildings with gaps and monthly totals, and a notebook charting the 4–8 PM peak (26% of daily use); all of it tested and running.
