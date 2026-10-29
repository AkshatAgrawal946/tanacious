# ⚡ Tanacious — Campus Energy Imputation Engine

> **AI Synergy Hackathon 2026** | **Case 05: Energy & Environment ("The 4 PM Spike")**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📖 Overview
**Tanacious** is a Data Integration & Load Imputation Engine designed to solve a fundamental energy blind spot for educational campuses. 

Campuses often experience a sharp electricity demand spike between 4 PM and 8 PM on weekdays, precisely when rooftop solar generation begins to drop[cite: 1]. Facilities teams cannot safely shift or optimize this load due to a massive data gap: some buildings report granular 15-minute interval data, while others only report aggregate monthly utility bills[cite: 1]. 

*You cannot optimize what you cannot measure.*[cite: 1] Tanacious bridges this gap using machine learning to infer continuous 15-minute load curves for unmetered buildings, delivering a unified, real-time campus demand profile[cite: 1].

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
