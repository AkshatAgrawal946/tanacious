from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import ml_model
from src.forecast import completed_bills, generate_forecast
from src.generate_data import generate
from src.overrides import Override, apply_overrides
from src.reconstruct import reconstruct_all


@pytest.fixture(scope="module")
def campus(tmp_path_factory):
    interval, bills, _ = generate(tmp_path_factory.mktemp("raw"))
    return interval, bills.assign(month=bills["month"].astype(str)), ml_model.train(interval, bills)


@pytest.fixture(scope="module")
def rec(campus):
    interval, bills, model = campus
    return reconstruct_all(interval, bills, ml_model.shape_fn(model, interval, bills))


def test_forecast_covers_48_hours_for_every_building(campus):
    interval, bills, model = campus
    by_building, total = generate_forecast(interval, bills, model, pd.Timestamp("2026-10-30"))
    assert len(total) == 48 * 4
    assert set(by_building["building"]) == set(bills["building"])
    assert (total["kW_low"] <= total["kW_high"]).all() and (by_building["kW"] >= 0).all()
    np.testing.assert_allclose(total["kW"], by_building.groupby("timestamp")["kW"].sum().to_numpy())


def test_forecast_uses_only_completed_bills(campus):
    _, bills, _ = campus
    known = completed_bills(bills, pd.Timestamp("2026-10-30"))
    assert known["month"].max() == "2026-09"


def test_override_halves_only_imputed_intervals_and_keeps_bill(rec):
    building = rec.loc[rec["is_observed"], "building"].iloc[0]  # a building with real readings too
    out = apply_overrides(rec, [Override(building, date(2026, 3, 10), date(2026, 3, 11))])
    hit = out["is_overridden"]
    assert hit.any() and not (hit & out["is_observed"]).any()
    np.testing.assert_allclose(out.loc[hit, "kWh_consumed"], out.loc[hit, "kWh_model"] * 0.5)
    observed = out["is_observed"]
    np.testing.assert_array_equal(out.loc[observed, "kWh_consumed"], rec.loc[observed, "kWh_consumed"])
    month = out[(out["building"] == building) & (out["month"] == "2026-03")]
    assert month["kWh_consumed"].sum() == pytest.approx(month["kWh_model"].sum())


def test_override_without_bill_matching_removes_energy(rec):
    building = rec["building"].iloc[0]
    out = apply_overrides(rec, [Override(building, date(2026, 3, 10), date(2026, 3, 10))], keep_bill=False)
    removed = out["kWh_model"].sum() - out["kWh_consumed"].sum()
    assert removed == pytest.approx(out.loc[out["is_overridden"], "kWh_model"].sum() * 0.5)


def test_no_overrides_is_a_no_op(rec):
    out = apply_overrides(rec, [])
    np.testing.assert_array_equal(out["kWh_consumed"], rec["kWh_consumed"])
    assert not out["is_overridden"].any()
