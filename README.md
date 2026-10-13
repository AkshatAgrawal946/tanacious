# ⚡ The Missing Meter: Prototype

> **AI Synergy Hackathon 2026** | **Theme:** Energy & Environment (The 4 PM Spike)

## 📖 Overview
This repository contains the baseline data integration and analysis layer for the "Missing Meter" problem. It provides a controlled test bed for generating synthetic campus energy records, mimicking incomplete monthly feeds, and performing Exploratory Data Analysis (EDA) to visualize the critical 4 PM to 8 PM demand spike before applying predictive models.

## ⚙️ Current Tech Stack
* **Data Processing & Analysis:** Python, Pandas, NumPy
* **Visualization:** Matplotlib, Jupyter Notebooks

## 🚀 What is Implemented Now
* **Synthetic Data Generator (`src/generate_data.py`):** Generates simulated hourly kWh readings for campus buildings, incorporating random multi-day gaps to mimic "monthly-only" unmetered constraints.
* **Baseline EDA (`notebooks/01_baseline_eda.ipynb`):** Resamples time-series data and isolates the 4 PM – 8 PM window to provide a demand-spike chart.

## 🔜 Proposed Next Steps (Round 2 Build)
* **Machine Learning Imputation:** Train and validate a regression model (Scikit-Learn/XGBoost) to infer the missing 15-minute intervals, constrained by known billing totals.
* **Interactive Dashboard:** Build a Streamlit interface to give facilities teams a verified 4–8 PM building-level demand profile for targeted planning.

---

**Done by:** Tejas S, Nikunj Negi, Akshat Agrawal, Adnan Bim Amjad
