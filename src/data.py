"""Input validation and data loading."""
from pathlib import Path

import pandas as pd

INTERVAL_COLUMNS = {"timestamp", "building", "building_type", "temperature", "kWh_consumed"}
BILL_COLUMNS = {"building", "building_type", "month", "monthly_kWh"}


def validate_inputs(interval: pd.DataFrame, bills: pd.DataFrame) -> list[str]:
    """Return human-readable data quality issues; an empty list is valid."""
    issues: list[str] = []
    for label, frame, required in [("interval", interval, INTERVAL_COLUMNS), ("monthly", bills, BILL_COLUMNS)]:
        missing = required - set(frame.columns)
        if missing:
            issues.append(f"{label} data is missing columns: {', '.join(sorted(missing))}")
    if issues:
        return issues
    parsed = pd.to_datetime(interval["timestamp"], errors="coerce")
    if parsed.isna().any():
        issues.append("interval data has unparseable timestamps")
    if interval.duplicated(["building", "timestamp"]).any():
        issues.append("interval data contains duplicate building/timestamp rows")
    if (bills["monthly_kWh"] < 0).any() or (interval["kWh_consumed"].dropna() < 0).any():
        issues.append("energy values cannot be negative")
    if bills.duplicated(["building", "month"]).any():
        issues.append("monthly data contains duplicate building/month rows")
    return issues


def load_data(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(data_dir)
    interval = pd.read_csv(path / "interval_15min.csv", parse_dates=["timestamp"])
    bills = pd.read_csv(path / "monthly_totals.csv")
    issues = validate_inputs(interval, bills)
    if issues:
        raise ValueError("; ".join(issues))
    bills["month"] = bills["month"].astype(str)
    return interval, bills
