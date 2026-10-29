import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, load_data, validation_report


def test_validator_reports_required_columns():
    issues = validation_report(pd.DataFrame(), pd.DataFrame()).errors
    assert len(issues) == 2
    assert "interval data is missing columns" in issues[0]


def _campus(days: int = 2, buildings=(("B1", "hostel"),)) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2026-03-01", periods=days * 96, freq="15min")
    interval = pd.concat([pd.DataFrame({"timestamp": ts, "building": b, "building_type": t, "temperature": 25.0,
                                        "kWh_consumed": 1.0}) for b, t in buildings], ignore_index=True)
    bills = pd.DataFrame([{"building": b, "building_type": t, "month": "2026-03", "monthly_kWh": 31 * 96.0}
                          for b, t in buildings])
    return interval, bills


def _errors(interval, bills) -> str:
    return " | ".join(validation_report(interval, bills).errors)


def test_clean_data_has_no_errors():
    assert validation_report(*_campus()).ok


@pytest.mark.parametrize("mutate, message", [
    (lambda i, b: i.assign(timestamp=i["timestamp"].astype(str).where(i.index != 3, "not a time")), "unparseable timestamps"),
    (lambda i, b: i.assign(timestamp=i["timestamp"].where(i.index != 5, pd.Timestamp("2026-03-01 00:07"))), "not aligned"),
    (lambda i, b: pd.concat([i, i.iloc[[0]]]), "duplicate building/timestamp"),
    (lambda i, b: i.assign(kWh_consumed=i["kWh_consumed"].where(i.index != 2, -1.0)), "negative kWh"),
    (lambda i, b: i.assign(kWh_consumed=i["kWh_consumed"].where(i.index != 2, np.inf)), "infinite"),
    (lambda i, b: i.assign(kWh_consumed=i["kWh_consumed"].astype(object).where(i.index != 2, "abc")), "non-numeric"),
    (lambda i, b: i.assign(building=i["building"].where(i.index != 1, "")), "blank building"),
])
def test_interval_failure_modes(mutate, message):
    interval, bills = _campus()
    assert message in _errors(mutate(interval, bills), bills)


@pytest.mark.parametrize("mutate, message", [
    (lambda b: pd.concat([b, b]), "duplicate building/month"),
    (lambda b: b.assign(monthly_kWh=-5.0), "negative monthly_kWh"),
    (lambda b: b.assign(month="March"), "malformed months"),
    (lambda b: b.assign(monthly_kWh="lots"), "non-numeric"),
    (lambda b: b.assign(building_type="lab"), "differs between interval and monthly"),
])
def test_bill_failure_modes(mutate, message):
    interval, bills = _campus()
    assert message in _errors(interval, mutate(bills))


def test_inconsistent_building_type_within_interval_data():
    interval, bills = _campus()
    interval.loc[0, "building_type"] = "lab"
    assert "more than one building_type" in _errors(interval, bills)


def test_interval_building_without_bills_is_an_error():
    interval, bills = _campus(buildings=(("B1", "hostel"), ("B2", "lab")))
    assert "no monthly bills: B2" in _errors(interval, bills[bills.building == "B1"])


def test_metered_energy_above_bill_is_an_error():
    interval, bills = _campus()
    assert "exceeds the bill" in _errors(interval, bills.assign(monthly_kWh=10.0))


def test_fully_metered_month_must_match_bill():
    ts = pd.date_range("2026-02-01", "2026-02-28 23:45", freq="15min")
    interval = pd.DataFrame({"timestamp": ts, "building": "B1", "building_type": "lab", "temperature": 20.0, "kWh_consumed": 1.0})
    bills = pd.DataFrame([{"building": "B1", "building_type": "lab", "month": "2026-02", "monthly_kWh": len(ts) + 50.0}])
    assert "every interval is metered" in _errors(interval, bills)
    assert validation_report(interval, bills.assign(monthly_kWh=float(len(ts)))).ok


def test_missing_readings_are_warnings_not_errors():
    interval, bills = _campus()
    interval.loc[10:20, "kWh_consumed"] = np.nan
    interval = interval.drop(index=range(40, 50))
    report = validation_report(interval, bills)
    assert report.ok
    assert any("11 blank readings and 10 absent" in w for w in report.warnings)


def test_missing_bill_month_is_a_warning():
    interval, bills = _campus()
    bills = pd.concat([bills, bills.assign(month="2026-05")])
    report = validation_report(interval, bills)
    assert report.ok and any("no bill for 2026-04" in w for w in report.warnings)


def test_timezone_aware_timestamps_are_rejected():
    interval, bills = _campus()
    interval["timestamp"] = interval["timestamp"].dt.tz_localize("Asia/Kolkata")
    assert "timezone" in _errors(interval, bills)


def test_load_data_raises_actionable_error(tmp_path):
    interval, bills = _campus()
    interval.loc[0, "kWh_consumed"] = -2
    interval.to_csv(tmp_path / "interval_15min.csv", index=False)
    bills.to_csv(tmp_path / "monthly_totals.csv", index=False)
    with pytest.raises(DataValidationError, match="negative kWh readings, e.g. B1 @ 2026-03-01"):
        load_data(tmp_path)


def test_load_data_never_reads_ground_truth(tmp_path):
    interval, bills = _campus()
    interval.to_csv(tmp_path / "interval_15min.csv", index=False)
    bills.to_csv(tmp_path / "monthly_totals.csv", index=False)
    (tmp_path / "ground_truth_15min.csv").write_text("corrupt, would fail if read")
    loaded, _ = load_data(tmp_path)
    assert len(loaded) == len(interval)
