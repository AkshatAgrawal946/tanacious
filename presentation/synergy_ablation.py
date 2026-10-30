"""Proof of synergy: Human only vs AI only vs Human + AI (plus the flat bill split).

Human only = facilities schedule estimates (TYPE_SCHEDULES x weekend x break), scaled to the bill.
AI only    = the same Random Forest on weather, calendar and clock time, without the human features.
Human + AI = the shipped model (src.ml_model).

Run from the repository root:
  PYTHONPATH=. python presentation/synergy_ablation.py synthetic   # 5 synthetic buildings
  PYTHONPATH=. python presentation/synergy_ablation.py 0.3         # real BDG2 meters, 30% of days hidden
  PYTHONPATH=. python presentation/synergy_ablation.py 1.0         # real BDG2 meters, fully unmetered
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src import ml_model, backtest
from src.ml_model import build_features, break_ratios, campus_temperature, is_break, HYPERPARAMETERS, SEED, TYPE_CALENDAR
from src.reconstruct import reconstruct_building_month
from src.evaluate import flat_shape
from src.metrics import score_prediction

AI_FEATURES = ["Is_Weekend", "Is_Break", "Temperature", "Temperature_3h_Mean", "Cooling_Degrees", "Hour"]

def feats(ts, btype, temp, ratio):
    f = build_features(ts, btype, temp, ratio)
    ts = pd.DatetimeIndex(ts); f["Hour"] = ts.hour + ts.minute / 60
    return f

def ai_train(interval, bills):
    # same rows/target as the real model, data-only features
    temperature = campus_temperature(interval); ratios = break_ratios(bills)
    X, y = [], []
    for b, rows in interval.dropna(subset=["kWh_consumed"]).groupby("building"):
        rows = rows.sort_values("timestamp"); ts = pd.DatetimeIndex(rows["timestamp"])
        own = bills[bills["building"] == b]
        n = sum(pd.Period(str(m), freq="M").days_in_month * 96 for m in own["month"])
        rel = rows["kWh_consumed"].to_numpy() / (own["monthly_kWh"].sum() / n)
        X.append(feats(ts, rows["building_type"].iloc[0], temperature, ratios[b])); y.append(rel / np.where(is_break(ts), ratios[b], 1.0))
    m = RandomForestRegressor(**HYPERPARAMETERS, n_jobs=-1, random_state=SEED)
    m.fit(pd.concat(X, ignore_index=True)[AI_FEATURES], np.concatenate(y))
    def predict(ts, b, btype):
        r = ratios.get(b, TYPE_CALENDAR.get(btype, (1, 1))[1])
        return np.clip(m.predict(feats(ts, btype, temperature, r)[AI_FEATURES]) * np.where(is_break(ts), r, 1.0), 1e-6, None)
    return predict

def human_shape(interval, bills):
    temperature = campus_temperature(interval); ratios = break_ratios(bills)
    def predict(ts, b, btype):
        r = ratios.get(b, TYPE_CALENDAR.get(btype, (1, 1))[1])
        return np.clip(build_features(ts, btype, temperature, r)["Expected_Occupancy"].to_numpy(), 1e-6, None)
    return predict

def shapes(reference, bills):
    return {"flat": flat_shape, "human": human_shape(reference, bills), "ai": ai_train(reference, bills),
            "human_ai": ml_model.shape_fn(ml_model.train(reference, bills), reference, bills)}

def rebuild(reference, bills, b, fn):
    months = sorted(bills.loc[bills["building"] == b, "month"].astype(str))
    return pd.concat([reconstruct_building_month(reference, bills, b, m, fn) for m in months]).set_index("timestamp")["kWh_consumed"]

def sc(actual, pred):
    idx = actual.dropna().index.intersection(pred.index)
    s = score_prediction(actual[idx].reset_index(drop=True), pred[idx].reset_index(drop=True), pd.Series(idx))
    peak = (idx.hour >= 16) & (idx.hour < 20)
    return s["wape_pct"], 100 * s["peak_window_error_kwh"] / actual[idx][peak].sum()

rows = []
which = sys.argv[1]
if which == "synthetic":
    from src.data import load_data
    from src.evaluate import load_ground_truth
    interval, bills = load_data(); bills = bills.assign(month=bills["month"].astype(str)); hidden = load_ground_truth()
    full = shapes(interval, bills)
    for b in sorted(bills["building"].unique()):
        if b in set(hidden["building"]):
            obs = set(interval.loc[(interval.building == b) & interval.kWh_consumed.notna(), "timestamp"])
            actual = hidden[hidden.building == b].set_index("timestamp")["kWh_consumed"]
            actual = actual[~actual.index.isin(obs)]; ref, fns = interval, full
        else:
            actual = interval[interval.building == b].set_index("timestamp")["kWh_consumed"]
            ref = interval[interval.building != b]; fns = shapes(ref, bills)
        for name, fn in fns.items():
            w, p = sc(actual, rebuild(ref, bills, b, fn)); rows.append((b, name, w, p))
else:
    raw = pd.read_csv("data/real/fox_campus_2017.csv")
    g = backtest.guess_columns(raw)
    data = backtest.prepare(raw, g["timestamp"], g["meters"], backtest.KWH, temperature=g["temperature"])
    types = {m: backtest.guess_type(m) for m in data.readings["building"].unique()}
    r = data.readings.assign(building_type=data.readings["building"].map(types))
    month = r["timestamp"].dt.to_period("M").astype(str).rename("month")
    bills = r.groupby(["building", "building_type", month])["kWh_consumed"].sum().rename("monthly_kWh").reset_index()
    share = float(which)  # 0.3 = 30% days hidden; 1.0 = whole building unmetered
    for b in sorted(types):
        own = (r.building == b).to_numpy()
        hid = np.zeros(len(r), bool)
        hid[own] = backtest.hide_random_days(r.loc[own, "timestamp"], share) if share < 1 else True
        ref = r.assign(kWh_consumed=r["kWh_consumed"].mask(hid | r["is_filled"].to_numpy()))[["timestamp","building","building_type","temperature","kWh_consumed"]]
        mask = hid & ~r["is_filled"].to_numpy()
        actual = r[mask].set_index("timestamp")["kWh_consumed"]
        for name, fn in shapes(ref, bills).items():
            pred = rebuild(ref, bills, b, fn)
            # hourly scoring like backtest.score
            a = actual.groupby(actual.index.floor("h")).sum(); pr = pred.reindex(actual.index).groupby(actual.index.floor("h")).sum()
            w, p = sc(a, pr); rows.append((b, name, w, p))
df = pd.DataFrame(rows, columns=["building", "arm", "wape", "peak_err"])
print(df.pivot(index="building", columns="arm", values="wape").round(1)[["flat","human","ai","human_ai"]].to_string())
print(df.pivot(index="building", columns="arm", values="peak_err").round(1)[["flat","human","ai","human_ai"]].to_string())
print("MEAN WAPE", df.groupby("arm")["wape"].mean().round(2).to_dict())
print("MEAN PEAK", df.groupby("arm")["peak_err"].mean().round(2).to_dict())
