import numpy as np
import pandas as pd
import pytest

from src.simulate import Scenario, Tariff, cost, fill_valleys, in_window, simulate


def _load(days=3, buildings=("B1", "B2")) -> pd.DataFrame:
    """A 20 kW floor with a 4-8 PM hump (up to 80 kW) on top, per building."""
    ts = pd.date_range("2026-03-01", periods=days * 96, freq="15min")
    hour = ts.hour + ts.minute / 60
    kw = 20 + 60 * np.clip(1 - np.abs(hour - 18) / 4, 0, None) + 5 * np.sin(hour)
    return pd.concat([pd.DataFrame({"timestamp": ts, "building": b, "kW": kw * (i + 1)})
                      for i, b in enumerate(buildings)], ignore_index=True)


def test_shifting_moves_energy_out_of_the_peak_and_charges_the_loss():
    load = _load()
    out = simulate(load, Scenario(shift_share=0.5, shift_loss=0.1))
    peak = in_window(out["timestamp"])
    moved = (out.loc[peak, "kW"] - out.loc[peak, "kW_after"]).sum()
    assert moved > 0
    assert np.isclose(out["kW_after"].sum() - out["kW"].sum(), 0.1 * moved)
    assert out.groupby("building").apply(lambda g: (g["kW_after"].max() < g["kW"].max())).all()


def test_base_load_is_never_touched_and_nothing_goes_negative():
    out = simulate(_load(), Scenario(shift_share=1.0, trim=1.0, shift_loss=0.0))
    assert (out["kW_after"] >= np.minimum(out["kW"], out["base_kW"]) - 1e-9).all()
    peak = in_window(out["timestamp"])
    assert np.allclose(out.loc[peak, "kW_after"], np.minimum(out.loc[peak, "kW"], out.loc[peak, "base_kW"]))


def test_trim_alone_scales_only_the_flexible_load():
    out = simulate(_load(), Scenario(shift_share=0.0, trim=0.2))
    flexible = (out["kW"] - out["base_kW"]).clip(lower=0)
    assert np.allclose(out["kW"] - out["kW_after"], 0.2 * flexible)


def test_a_null_scenario_changes_nothing_and_each_day_keeps_its_energy():
    load = _load()
    assert np.allclose(simulate(load, Scenario(shift_share=0.0))["kW_after"], load["kW"])
    out = simulate(load, Scenario(shift_share=0.4, shift_loss=0.0))
    daily = out.groupby([out["building"], out["timestamp"].dt.date])[["kW", "kW_after"]].sum()
    assert np.allclose(daily["kW"], daily["kW_after"])


def test_valleys_fill_lowest_first_to_one_level():
    add = fill_valleys(np.array([[1.0, 5.0, 2.0, 9.0], [3.0, 3.0, 3.0, 3.0]]), np.array([5.0, 0.0]))
    assert np.allclose(add[0], [3.0, 0.0, 2.0, 0.0])  # 1 and 2 both rise to 4, 5 and 9 untouched
    assert np.allclose(add[1], 0.0)
    huge = fill_valleys(np.array([[1.0, 3.0]]), np.array([10.0]))
    assert np.allclose(huge[0] + [1.0, 3.0], 7.0)


def test_bad_scenarios_and_gappy_grids_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        simulate(_load(), Scenario(target_hours=(14, 18)))
    with pytest.raises(ValueError, match="empty"):
        simulate(_load(), Scenario(target_hours=(12, 12)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate(_load(), Scenario(shift_share=1.5))
    with pytest.raises(ValueError, match="full 15-minute grid"):
        simulate(_load().drop(index=5), Scenario())


def test_cost_prices_peak_energy_and_monthly_maximum_demand():
    ts = pd.Series(pd.to_datetime(["2026-03-01 10:00", "2026-03-01 17:00", "2026-04-01 17:00"]))
    bill = cost(ts, pd.Series(["B1"] * 3), pd.Series([4.0, 8.0, 12.0]), Tariff(off_peak=1, peak=2, demand=10))
    assert np.isclose(bill.loc["B1", "energy"], 1 * 1 + 2 * 2 + 3 * 2)
    assert np.isclose(bill.loc["B1", "demand"], (8 + 12) * 10)


def test_moved_load_fills_campus_valleys_and_bystanders_are_untouched():
    load = _load()
    # B2 is busy at 2-4 PM, so the moved load should land at 12-2 PM instead.
    busy = in_window(load["timestamp"], (14, 16)) & (load["building"] == "B2")
    load.loc[busy, "kW"] += 200
    out = simulate(load, Scenario(shift_share=0.3, target_hours=(12, 16), shift_loss=0.0), taking_part=["B1"])
    b1, b2 = out[out["building"] == "B1"], out[out["building"] == "B2"]
    assert np.allclose(b2["kW_after"], b2["kW"]) and b2["base_kW"].isna().all()
    gained = (b1["kW_after"] - b1["kW"]).clip(lower=0)
    assert gained[in_window(b1["timestamp"], (12, 14))].sum() > 0
    assert np.isclose(gained[in_window(b1["timestamp"], (14, 16))].sum(), 0)
    campus = out.groupby("timestamp")[["kW", "kW_after"]].sum()
    assert campus["kW_after"].max() <= campus["kW"].max()

