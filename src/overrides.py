"""Human-in-the-loop context overrides (INTERVENE: administrators correct what the model cannot know).

The forest learns the timetable and the academic calendar, but not a snap holiday, a building
closure or an exam week. An override scales the reconstructed load on chosen dates by an
occupancy factor (0.5 = "half the building is in").

Two rules keep an override honest:

* Only AI-imputed intervals change. Real meter readings are measurements, not guesses.
* With ``keep_bill=True`` (the default) the energy taken off the override dates is handed back to
  the month's other imputed intervals, so the month still matches its bill. The bill is a fact; the
  override corrects *when* that energy was used, not *how much*. With ``keep_bill=False`` the cut is
  applied as-is and the month falls below its bill.
"""
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Override:
    building: str
    start: date
    end: date  # inclusive
    factor: float = 0.5
    note: str = "Holiday / 50% occupancy"


def apply_overrides(rec: pd.DataFrame, overrides: list[Override], keep_bill: bool = True) -> pd.DataFrame:
    """``rec`` (from ``reconstruct_all``) with every override applied. Adds ``kWh_model`` (the
    model's original value) and ``is_overridden``; recomputes ``kW``."""
    out = rec.assign(kWh_model=rec["kWh_consumed"], is_overridden=False)
    day = out["timestamp"].dt.normalize()
    for o in overrides:
        hit = ((out["building"] == o.building) & day.between(pd.Timestamp(o.start), pd.Timestamp(o.end))
               & out["is_reconstructed"])
        out.loc[hit, "kWh_consumed"] *= o.factor
        out.loc[hit, "is_overridden"] = True
    if keep_bill and overrides:
        # Rescale the untouched imputed intervals of each affected month so imputed + metered
        # energy still equals the bill. If every imputed interval is overridden there is nothing
        # left to absorb the difference, and the month is left below its bill.
        for (building, month), rows in out[out["is_overridden"]].groupby(["building", "month"]):
            in_month = (out["building"] == building) & (out["month"] == month)
            free = in_month & out["is_reconstructed"] & ~out["is_overridden"]
            target = out.loc[in_month, "kWh_model"].sum() - out.loc[in_month & ~free, "kWh_consumed"].sum()
            current = out.loc[free, "kWh_consumed"].sum()
            if free.any() and current > 0:
                out.loc[free, "kWh_consumed"] *= target / current
    return out.assign(kW=out["kWh_consumed"] * 4)
