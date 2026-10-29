"""Input validation and data loading.

Data contract (see README "Data contracts"):

* ``interval_15min.csv``: ``timestamp, building, building_type, temperature, kWh_consumed``.
  Timestamps are timezone-naive campus local time on a 15-minute grid (no DST shifts). A blank
  ``kWh_consumed`` is a legitimate missing reading (sensor dropout) and is reported as a warning,
  not an error. A building may be metered for only part of the year (e.g. a pilot meter).
* ``monthly_totals.csv``: ``building, building_type, month (YYYY-MM), monthly_kWh``; exactly one
  bill per building-month.

Errors make the data unusable (``load_data`` raises); warnings are surfaced but allowed.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INTERVAL_COLUMNS = {"timestamp", "building", "building_type", "temperature", "kWh_consumed"}
BILL_COLUMNS = {"building", "building_type", "month", "monthly_kWh"}
FREQ = pd.Timedelta("15min")
# Metered energy in a month may differ from its bill by at most this much (rounding of the bill and
# of each reading). ``reconstruct.py`` uses the same tolerance for its accounting checks.
BILL_TOLERANCE_KWH = 0.01


class DataValidationError(ValueError):
    """Raised by ``load_data`` when the inputs break the data contract."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _examples(values, limit: int = 3) -> str:
    values = [str(v) for v in values]
    return ", ".join(values[:limit]) + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


def _parse_timestamps(raw: pd.Series, report: ValidationReport) -> pd.Series | None:
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
    except (ValueError, TypeError) as error:  # e.g. mixed UTC offsets
        report.errors.append(f"interval timestamps could not be parsed ({error}); use timezone-naive campus local time")
        return None
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        report.errors.append("interval timestamps carry a timezone; convert them to timezone-naive campus local time")
        return None
    bad = raw[parsed.isna()]
    if len(bad):
        report.errors.append(f"interval data has {len(bad)} unparseable timestamps, e.g. {_examples(bad.astype(str).unique())}")
    return parsed


def _numeric(frame: pd.DataFrame, column: str, label: str, report: ValidationReport) -> pd.Series:
    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    malformed = raw[values.isna() & raw.notna()]
    if len(malformed):
        report.errors.append(f"{label} {column} has {len(malformed)} non-numeric values, e.g. {_examples(malformed.unique())}")
    infinite = np.isinf(values)
    if infinite.any():
        report.errors.append(f"{label} {column} has {int(infinite.sum())} infinite values")
    return values.where(~infinite)


def _check_interval(interval: pd.DataFrame, report: ValidationReport) -> pd.DataFrame | None:
    parsed = _parse_timestamps(interval["timestamp"], report)
    if parsed is None:
        return None
    frame = interval.assign(timestamp=parsed)[parsed.notna()]
    if frame["building"].isna().any() or (frame["building"].astype(str).str.strip() == "").any():
        report.errors.append("interval data has rows with a blank building identifier")
    off_grid = frame.loc[frame["timestamp"] != frame["timestamp"].dt.floor(FREQ), "timestamp"]
    if len(off_grid):
        report.errors.append(f"{len(off_grid)} interval timestamps are not aligned to the 15-minute grid, "
                             f"e.g. {_examples(off_grid.astype(str).unique())}")
    dupes = frame[frame.duplicated(["building", "timestamp"], keep=False)]
    if len(dupes):
        pairs = (dupes["building"].astype(str) + " @ " + dupes["timestamp"].astype(str)).unique()
        report.errors.append(f"interval data contains {len(pairs)} duplicate building/timestamp rows, e.g. {_examples(pairs)}")

    frame = frame.assign(kWh_consumed=_numeric(frame, "kWh_consumed", "interval", report),
                         temperature=_numeric(frame, "temperature", "interval", report))
    negative = frame[frame["kWh_consumed"] < 0]
    if len(negative):
        report.errors.append("energy values cannot be negative: interval data has "
                             f"{len(negative)} negative kWh readings, e.g. "
                             f"{_examples(negative['building'].astype(str) + ' @ ' + negative['timestamp'].astype(str))}")

    for building, rows in frame.groupby("building"):
        missing = int(rows["kWh_consumed"].isna().sum())
        span = (rows["timestamp"].max() - rows["timestamp"].min()) // FREQ + 1
        absent = int(span - rows["timestamp"].nunique())
        if missing or absent:
            report.warnings.append(f"{building}: {missing} blank readings and {absent} absent 15-minute rows inside "
                                   "its metered span; these intervals are treated as missing")
    if frame["temperature"].isna().any():
        report.warnings.append(f"{int(frame['temperature'].isna().sum())} interval rows have no temperature; "
                               "it is interpolated from neighbouring readings")
    return frame


def _check_bills(bills: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    frame = bills.assign(month=bills["month"].astype(str))
    periods = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    bad = frame.loc[periods.isna(), "month"]
    if len(bad):
        report.errors.append(f"monthly data has malformed months (expected YYYY-MM), e.g. {_examples(bad.unique())}")
    values = _numeric(frame, "monthly_kWh", "monthly", report)
    if values.isna().any() and not report.errors:
        report.errors.append(f"monthly data has {int(values.isna().sum())} blank monthly_kWh values")
    if (values < 0).any():
        report.errors.append("energy values cannot be negative: monthly data has negative monthly_kWh")
    frame = frame.assign(monthly_kWh=values)
    dupes = frame[frame.duplicated(["building", "month"], keep=False)]
    if len(dupes):
        report.errors.append("monthly data contains duplicate building/month rows: "
                             + _examples((dupes["building"].astype(str) + " " + dupes["month"]).unique()))
    frame = frame[periods.notna()]
    for building, rows in frame.groupby("building"):
        months = pd.PeriodIndex(rows["month"].unique(), freq="M").sort_values()
        gaps = pd.period_range(months.min(), months.max(), freq="M").difference(months)
        if len(gaps):
            report.warnings.append(f"{building} has no bill for {_examples(gaps.astype(str))}; those months cannot be reconstructed")
    return frame


def _check_consistency(interval: pd.DataFrame, bills: pd.DataFrame, report: ValidationReport,
                       tolerance_kwh: float) -> None:
    for label, frame in (("interval", interval), ("monthly", bills)):
        types = frame.groupby("building")["building_type"].nunique()
        if (types > 1).any():
            report.errors.append(f"{label} data gives more than one building_type for {_examples(types[types > 1].index)}")
    interval_types = interval.groupby("building")["building_type"].first()
    bill_types = bills.groupby("building")["building_type"].first()
    shared = interval_types.index.intersection(bill_types.index)
    mismatched = shared[interval_types[shared].to_numpy() != bill_types[shared].to_numpy()]
    if len(mismatched):
        report.errors.append(f"building_type differs between interval and monthly data for {_examples(mismatched)}")
    unbilled = interval_types.index.difference(bill_types.index)
    if len(unbilled):
        report.errors.append(f"interval data has buildings with no monthly bills: {_examples(unbilled)}")

    observed = interval.dropna(subset=["kWh_consumed"])
    metered = (observed.assign(month=observed["timestamp"].dt.to_period("M").astype(str))
               .groupby(["building", "month"])["kWh_consumed"].agg(["sum", "count"]))
    billed = bills.drop_duplicates(["building", "month"]).set_index(["building", "month"])["monthly_kWh"]
    joined = metered.join(billed, how="inner")
    if len(joined):
        days = pd.PeriodIndex(joined.index.get_level_values("month"), freq="M").days_in_month.to_numpy()
        over = joined["sum"] > joined["monthly_kWh"] + tolerance_kwh
        complete = (joined["count"].to_numpy() == days * 96) & ((joined["sum"] - joined["monthly_kWh"]).abs() > tolerance_kwh)
        for (building, month), row in joined[over].iterrows():
            report.errors.append(f"{building} {month}: metered energy {row['sum']:.2f} kWh exceeds the bill "
                                 f"{row['monthly_kWh']:.2f} kWh by more than {tolerance_kwh} kWh")
        for (building, month), row in joined[complete & ~over].iterrows():
            report.errors.append(f"{building} {month}: every interval is metered but the readings sum to "
                                 f"{row['sum']:.2f} kWh, not the billed {row['monthly_kWh']:.2f} kWh")
    unmatched = metered.index.difference(billed.index)
    if len(unmatched):
        report.warnings.append(f"{len(unmatched)} metered building-months have no bill, e.g. "
                               f"{_examples(f'{b} {m}' for b, m in unmatched)}")


def validation_report(interval: pd.DataFrame, bills: pd.DataFrame,
                      tolerance_kwh: float = BILL_TOLERANCE_KWH) -> ValidationReport:
    """Check the inputs against the data contract; see the module docstring."""
    report = ValidationReport()
    for label, frame, required in [("interval", interval, INTERVAL_COLUMNS), ("monthly", bills, BILL_COLUMNS)]:
        missing = required - set(frame.columns)
        if missing:
            report.errors.append(f"{label} data is missing columns: {', '.join(sorted(missing))}")
    if report.errors:
        return report
    clean_interval = _check_interval(interval, report)
    clean_bills = _check_bills(bills, report)
    if clean_interval is not None and report.ok:
        _check_consistency(clean_interval, clean_bills, report, tolerance_kwh)
    return report


def load_data(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the operational inputs. The hidden ground-truth file is never read here:
    it is evaluation-only (``src.evaluate.load_ground_truth``)."""
    path = Path(data_dir)
    interval = pd.read_csv(path / "interval_15min.csv")
    bills = pd.read_csv(path / "monthly_totals.csv", dtype={"month": str})
    report = validation_report(interval, bills)
    for warning in report.warnings:
        log.warning("data warning: %s", warning)
    if report.errors:
        raise DataValidationError("; ".join(report.errors))
    interval["timestamp"] = pd.to_datetime(interval["timestamp"])
    interval["kWh_consumed"] = pd.to_numeric(interval["kWh_consumed"]).astype(float)
    return interval, bills
