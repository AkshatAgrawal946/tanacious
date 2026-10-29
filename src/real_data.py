"""Real meter data for the "Test on your data" tab: seven buildings of one US university campus.

Source: the Building Data Genome Project 2 (BDG2, https://github.com/buds-lab/building-data-genome-project-2),
hourly electricity of 1,636 real buildings for 2016-2017, CC BY-SA 4.0. We use site "Fox" (US/Mountain
time, which does not observe daylight saving; desert climate, 48 °C summer peak): classrooms, labs,
dormitories and a library, i.e. the model's academic, lab, hostel and library types.

``python -m src.real_data`` downloads BDG2's metadata, weather and cleaned electricity files (about
195 MB, checked against their SHA-256 hashes) into ``data/external/bdg2/`` (git-ignored), then cleans
one year for the chosen meters and writes ``data/real/fox_campus_<year>.csv``:

* a full hourly grid (a missing hour is a blank cell);
* readings that cannot be real are blanked, never replaced: negative values, a meter frozen on one
  value for ``STUCK_HOURS`` or more, a single-hour spike (above ``SPIKE_FACTOR`` times the median of
  the surrounding day and over twice both neighbours) and dropouts below ``1 / DROPOUT_FACTOR`` of
  that median. The bars are set wide on purpose: BDG2's "cleaned" files already drop obvious faults,
  and real buildings do odd-looking things on purpose (the library falls to a fifth of its daytime
  load at 4-5 AM every day, and opens for a few hours on otherwise closed summer Saturdays);
* outdoor temperature from the site's weather station, short gaps (up to 6 hours) interpolated.

The upload in the dashboard interpolates the few blanks left in a month and never scores them.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {  # file -> (path in the repository, SHA-256 from its Git LFS pointer)
    "metadata.csv": ("metadata/metadata.csv", "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"),
    "weather.csv": ("weather/weather.csv", "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd"),
    "electricity_cleaned.csv": ("meters/cleaned/electricity_cleaned.csv",
                                "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"),
}
SITE = "Fox"
# BDG2 meter -> column name in the export. The first word tells the dashboard the building type
# (Classroom -> academic, Lab -> lab, Dormitory -> hostel, Library -> library). Chosen for complete
# data and clearly different daily shapes.
METERS = {
    "Fox_education_Nilda": "Classroom Nilda kWh",
    "Fox_education_Long": "Classroom Long kWh",
    "Fox_education_Andre": "Lab Andre kWh",
    "Fox_education_Theodore": "Lab Theodore kWh",
    "Fox_lodging_Helen": "Dormitory Helen kWh",
    "Fox_lodging_Winifred": "Dormitory Winifred kWh",
    "Fox_public_Bart": "Library Bart kWh",
}
TEMPERATURE = "Outdoor temperature (C)"
STUCK_HOURS = 6  # a meter repeating one value this long is frozen, not measuring
SPIKE_FACTOR = 3.0  # a reading this many times its surrounding day's median is a glitch
DROPOUT_FACTOR = 10.0  # ...and so is one under a tenth of it (a dead meter or a lost reading)


def download(directory: Path) -> None:
    """Fetch the BDG2 files that are not already present, and check every file's SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (path, sha256) in FILES.items():
        target = directory / name
        if not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{path}", target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"{target} does not match the published file (SHA-256 {digest}); delete it and re-run")


def clean_meter(kwh: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Blank readings that cannot be real; return the cleaned series and how many of each were blanked."""
    counts = {"missing": int(kwh.isna().sum())}
    negative = kwh < 0
    run = kwh.ne(kwh.shift()).cumsum()  # consecutive equal values share a run id
    stuck = kwh.notna() & kwh.groupby(run).transform("size").ge(STUCK_HOURS)
    kept = kwh.mask(negative | stuck)
    day_median = kept.rolling(25, center=True, min_periods=12).median()
    neighbours = pd.concat([kept.shift(), kept.shift(-1)], axis=1).max(axis=1)
    spike = kept.notna() & (kept > SPIKE_FACTOR * day_median) & (kept > 2 * neighbours)
    dropout = kept.notna() & (kept < day_median / DROPOUT_FACTOR)
    counts.update(negative=int(negative.sum()), stuck=int(stuck.sum()), spike=int(spike.sum()), dropout=int(dropout.sum()))
    return kept.mask(spike | dropout), counts


def build(directory: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the cleaned hourly export, a per-meter cleaning report)."""
    grid = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    meta = pd.read_csv(directory / "metadata.csv").set_index("building_id")
    electricity = pd.read_csv(directory / "electricity_cleaned.csv", usecols=["timestamp", *METERS],
                              parse_dates=["timestamp"], index_col="timestamp")
    if electricity.index.duplicated().any():
        raise ValueError("repeated timestamps in the electricity file")
    weather = pd.read_csv(directory / "weather.csv", usecols=["timestamp", "site_id", "airTemperature"],
                          parse_dates=["timestamp"])
    temperature = weather[weather["site_id"] == SITE].set_index("timestamp")["airTemperature"]

    export = pd.DataFrame(index=grid)
    report = []
    for meter, column in METERS.items():
        cleaned, counts = clean_meter(electricity[meter].reindex(grid))
        export[column] = cleaned.round(2)
        info = meta.loc[meter]
        report.append({"column": column, "bdg2_meter": meter, "use": info["sub_primaryspaceusage"],
                       "floor_area_m2": info["sqm"], **counts, "blank_after_cleaning": int(cleaned.isna().sum()),
                       "mean_kWh": round(float(cleaned.mean()), 1)})
    export[TEMPERATURE] = temperature.reindex(grid).interpolate(limit=6, limit_area="inside").round(1)
    export.index.name = "Timestamp"
    return export, pd.DataFrame(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2017, choices=[2016, 2017])
    parser.add_argument("--cache", type=Path, default=Path("data/external/bdg2"))
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    download(args.cache)
    export, report = build(args.cache, args.year)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"fox_campus_{args.year}.csv"
    export.to_csv(path, date_format="%Y-%m-%d %H:%M")
    print(f"Wrote {path} ({len(export):,} hours x {len(METERS)} meters)")
    with pd.option_context("display.width", 200):
        print(report.to_string(index=False))
    blank = np.round(100 * report["blank_after_cleaning"].sum() / (len(export) * len(METERS)), 2)
    print(f"{blank}% of readings blank after cleaning")


if __name__ == "__main__":
    main()
