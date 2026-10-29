<div align="center">

# ⚡ Tanacious — Campus Energy

### *The Missing Meter: seeing the 4 PM spike in buildings that have no smart meter*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype_1-orange)
![Hackathon](https://img.shields.io/badge/AI_Synergy_Hackathon-2026-blueviolet)

</div>

---

## 📖 About

Our campus electricity demand jumps sharply between **4 PM and 8 PM on weekdays**, right when rooftop solar output starts to fall. The catch: some buildings report **15-minute meter data**, while others only have a **monthly utility bill total**. 📉

> 💡 *You cannot optimize or shift what you cannot measure.*

**Tanacious** is a prototype for the *AI Synergy Hackathon 2026* (Case 05, **"The 4 PM Spike"**, Energy & Environment). The idea is to fill in the missing meters using machine learning, so facilities teams can see where the evening peak really comes from.

---

## 🎯 The Problem

| 🏢 Building type | 📊 Data available | 😕 Challenge |
|---|---|---|
| Metered buildings | 15-minute kWh readings | Occasional sensor-dropout gaps |
| Unmetered buildings | Monthly bill totals only | Can't tell *when* they draw power |

A flat average of a monthly bill can't capture the evening surge, so the peak can't be attributed to a building or optimized.

---

## 🧠 Our Approach

1. 🔌 **Learn** load patterns from the buildings that *do* have 15-minute meters.
2. 🌦️ **Use context** such as hour of day, weekday or weekend, temperature, and building type.
3. 🧩 **Impute** continuous 15-minute load curves for unmetered buildings, constrained by their monthly totals.
4. 📈 **Compare** predictions against ground truth to prove the model works.
5. 🧑‍💼 **Support decisions**, with humans in the loop and **no autonomous shutdowns**.

Candidate models: Linear Regression, Random Forest, XGBoost.

---

## 🚀 Quick Start

```bash
# 1️⃣ Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Generate the synthetic campus dataset (writes data/raw/*.csv)
python src/generate_data.py

# 4️⃣ Explore the baseline analysis
jupyter notebook notebooks/01_baseline_eda.ipynb
```

---

## 🗂️ Project Structure

```text
tanacious/
├── 📁 data/                 # Generated datasets (data/raw/*.csv)
├── 📁 notebooks/
│   └── 01_baseline_eda.ipynb   # Resampling + 4–8 PM demand-spike chart
├── 📁 src/
│   └── generate_data.py        # Synthetic data generator
├── 📄 requirements.txt
├── 📄 Problem Statement Justifications.pdf
└── 📄 6abc0300be656_ai_synergy_hackathon_2026_round_1_case_file_redesigned.pdf
```

---

## 🧪 Synthetic Dataset

`src/generate_data.py` simulates **hourly-to-15-minute kWh data for 5 buildings** from **Jun to Aug 2026**, with random multi-day gaps. True monthly totals are saved separately.

- 🏠 Hostel A and Hostel B: monthly totals only
- 🏫 Academic block: 15-minute metered
- 📚 Library: monthly totals only
- 🔬 Lab: 15-minute metered

Each building follows a base load plus an evening peak, a cooling effect when it's hot, and lower weekend usage. 🌡️

> ⚠️ The hidden ground-truth series is for **evaluation only** and is never used as model input.

---

## 🗺️ Roadmap

- [x] 🧱 Synthetic data generator
- [x] 📓 Baseline EDA notebook with 4–8 PM spike chart
- [ ] 🤖 Load imputation model (Pandas + scikit-learn)
- [ ] 🚨 Missing-data fail-safes and flags
- [ ] 🖥️ Streamlit dashboard: predicted vs ground-truth load curves
- [ ] 🏷️ Peak attribution by zone and building type
- [ ] 🧑‍💼 Human-in-the-loop decision support

---

## 🛡️ Safety Boundaries

- 🚫 No direct control of electrical infrastructure.
- 🧪 Any simulated control is clearly labelled as a **simulation**.
- 🔒 Critical loads (labs, server rooms) are never automated.

---

<div align="center">

Made with ❤️ for the **AI Synergy Hackathon 2026**

</div>
