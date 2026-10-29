# Real meter data: `fox_campus_2017.csv`

Hourly electricity of seven buildings on one US university campus for 2017, plus outdoor temperature,
ready for the dashboard's **Test on your data** tab (upload it; every setting is guessed correctly).

## Source and licence

From the **Building Data Genome Project 2** (BDG2): Miller, C., Kathirgamanathan, A., Picchetti, B. et al.,
*The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III
competition*, Scientific Data 7, 368 (2020), https://doi.org/10.1038/s41597-020-00712-x.
Data: https://github.com/buds-lab/building-data-genome-project-2.

BDG2 is licensed under **CC BY-SA 4.0**, and so is this cleaned extract: credit BDG2 when you show or
share it, and share anything derived from it under the same licence.

## What is in it

Site "Fox": US/Mountain time without daylight saving (one reading per hour, no repeated or missing
clock hours), desert climate (12.7 °C January mean, 48.3 °C peak). Timestamps are local time: outdoor
temperature peaks at 4 PM, classrooms at midday to 3 PM, dormitories at 9 PM.

| Column | BDG2 meter | Use | Floor area | Model type |
|---|---|---|---|---|
| Classroom Nilda kWh | Fox_education_Nilda | College classroom | 10,371 m² | academic |
| Classroom Long kWh | Fox_education_Long | College classroom | 12,734 m² | academic |
| Lab Andre kWh | Fox_education_Andre | College laboratory | 17,044 m² | lab |
| Lab Theodore kWh | Fox_education_Theodore | College laboratory | 11,389 m² | lab |
| Dormitory Helen kWh | Fox_lodging_Helen | Dormitory | 51,395 m² | hostel |
| Dormitory Winifred kWh | Fox_lodging_Winifred | Dormitory | 5,582 m² | hostel |
| Library Bart kWh | Fox_public_Bart | Library | 9,408 m² | library |
| Outdoor temperature (C) | Fox weather station | | | |

Values are kWh used in each hour (the tab's "kWh per interval"). The meters were chosen for complete
data (every month at least 95% present in both years) and clearly different daily shapes.

## Cleaning (`python -m src.real_data`)

The script downloads BDG2's metadata, weather and *cleaned* electricity files, checks them against the
SHA-256 hashes published in the repository, and keeps 2017 for these meters (`--year 2016` for the
other year). Readings that cannot be real are **blanked, never replaced**:

- negative values: none found;
- a meter frozen on one value for 6 hours or more: none found;
- a single-hour spike, over 3x the median of the surrounding day and over twice both neighbours: none found;
- a dropout below a tenth of that median: 2 (e.g. Lab Andre read 22 kWh between 317 and 163).

With the 9 hours already missing in BDG2, 11 of 61,320 readings (0.02%) are blank. The tab
interpolates those and never scores them. Outdoor temperature gaps of up to 6 hours are interpolated.

The rules are deliberately loose: real buildings do things on purpose that look like faults. The
library drops to a fifth of its daytime load at 4-5 AM every day, and opens for a few hours on
otherwise closed summer Saturdays. A tighter filter blanked 8% of its year, all of it real.

## How the model does on it

Error (WAPE) on the hidden hours, model vs. the flat bill split (lower is better):

| Meter | 30% of days hidden | 11-24 Sep 2017 hidden |
|---|---|---|
| Classroom Nilda | 9.4% vs 21.2% | 6.0% vs 22.3% |
| Classroom Long | 13.4% vs 26.6% | 11.2% vs 27.4% |
| Lab Andre | 4.5% vs 8.2% | 3.9% vs 8.3% |
| Lab Theodore | 3.9% vs 3.5% | 2.6% vs 4.0% |
| Dormitory Helen | 9.2% vs 13.3% | 7.1% vs 13.1% |
| Dormitory Winifred | 5.9% vs 7.8% | 5.0% vs 8.0% |
| Library Bart | 21.4% vs 40.1% | 14.9% vs 40.0% |

Known gaps this data shows:

- **Holidays.** The model's academic calendar (`src/campus_calendar.py`) is the 2026 campus's, so it
  does not know this campus's breaks: on 2 January 2017 (a holiday) it predicts a normal teaching day.
- **The library peaks at midday**, while the model's library schedule expects the evening.
- **A nearly flat lab** (Theodore) is already well served by an even split of its bill.
