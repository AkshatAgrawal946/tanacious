"""What-if load shifting: how much of the evening peak could pre-cooling and setpoint changes remove?

Each building's 15-minute load is split into a flat base load (critical plug load, servers, lab
equipment, never touched) and the flexible, weather-driven load above it (HVAC). A scenario then

1. trims the flexible load everywhere (a setpoint raise, better controls), then
2. moves a share of the flexible load out of the peak window into an earlier window on the same
   day (pre-cooling), paying a loss on the energy moved (a cooled building leaks some back),
3. filling the campus's quietest intervals in that window first, so the moved load does not
   build a new peak.

This is a planning sketch on reconstructed load, not a controls model: it assumes the building
can store the moved cooling and that occupants accept the setpoint. Every day is shifted within
itself, so no energy moves between days, months or bills.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
PEAK_HOURS = (16, 20)  # 4-8 PM, the window the dashboard already highlights


@dataclass(frozen=True)
class Scenario:
    """Defaults are the dashboard's starting point, found by searching every share (5% steps) and every
    pre-cool window ending at 4 PM that starts no more than 6 hours earlier, over each month of the 2026
    synthetic data: the largest share that still gives the full cut in campus peak, with no month's peak
    rising. Beyond about 25% the campus peak moves after 8 PM (hostel evenings), so moving more only adds
    pre-cooling losses and, past ~35%, an afternoon peak. Windows further from the peak scored better only
    because the flat ``shift_loss`` ignores that a building cannot hold cooling for many hours.
    Re-run the search on real data before relying on them."""
    shift_share: float = 0.25         # share of flexible load moved out of the peak window
    target_hours: tuple[int, int] = (10, 16)  # where it goes: [start, end) hours of the same day
    shift_loss: float = 0.1           # extra energy needed per kWh moved
    trim: float = 0.0                 # share of flexible load removed everywhere
    peak_hours: tuple[int, int] = PEAK_HOURS


@dataclass(frozen=True)
class Tariff:
    """Time-of-use energy rates plus a demand charge on each building's monthly 15-minute maximum.
    Defaults are placeholders: replace them with the campus's own tariff."""
    off_peak: float = 7.0   # per kWh
    peak: float = 10.0      # per kWh inside the peak window
    demand: float = 300.0   # per kW of monthly maximum demand, per building


def base_load(kw: pd.Series, hour: pd.Series) -> float:
    """The floor a building never drops below: the 5th percentile of its 15-minute demand, capped
    at its average day's minimum so it is one flat line under the whole curve (hostels dip lowest
    mid-day, when students are out, not overnight)."""
    average_day = kw.groupby(hour).mean()
    return min(float(kw.quantile(0.05)), float(average_day.min()))


def _hours_mask(hours: tuple[int, int]) -> np.ndarray:
    slot_hour = np.arange(SLOTS_PER_DAY) / 4
    return (slot_hour >= hours[0]) & (slot_hour < hours[1])


def fill_valleys(load: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Per row, the additions that raise ``load``'s lowest values to one common level, using exactly
    ``energy`` (in the same units as ``load``) per row. Rows are days, columns are intervals."""
    load, energy = np.atleast_2d(load), np.asarray(energy, dtype=float)
    ordered = np.sort(load, axis=1)
    filled = np.cumsum(ordered, axis=1)
    count = np.arange(1, load.shape[1] + 1)
    # The j lowest intervals are raised when lifting them to the j-th lowest value costs less than
    # ``energy``; that cost only grows with j, so the count of such j is how many get filled.
    reached = ((count * ordered - filled) < energy[:, None]).sum(axis=1).clip(min=1)
    level = (energy + filled[np.arange(len(load)), reached - 1]) / reached
    return np.where(energy[:, None] > 0, np.clip(level[:, None] - load, 0, None), 0.0)


def _grid(rows: pd.DataFrame, building: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``rows``' kW as a (days, 96) array, plus each row's day number and slot of the day."""
    ts = rows["timestamp"]
    slot = (ts.dt.hour * 4 + ts.dt.minute // 15).to_numpy()
    days, day = np.unique(ts.dt.normalize().to_numpy(), return_inverse=True)
    cells = len(days) * SLOTS_PER_DAY
    if len(rows) != cells or np.unique(day * SLOTS_PER_DAY + slot).size != cells:
        raise ValueError(f"{building} is not on a full 15-minute grid")
    grid = np.empty((len(days), SLOTS_PER_DAY))
    grid[day, slot] = rows["kW"].to_numpy()
    return grid, day, slot


def simulate(load: pd.DataFrame, scenario: Scenario, taking_part=None) -> pd.DataFrame:
    """``load`` has ``timestamp``, ``building`` and ``kW`` on a full 15-minute grid. Returns it with
    ``base_kW`` and ``kW_after`` (``kW`` is left as it was). Only buildings in ``taking_part`` (default:
    all) change, but every building counts towards the campus load whose valleys are filled: the
    load moved each day is laid into the quietest intervals of the *campus* total, so buildings do
    not all pile into the same hours and build a new campus peak. Each building's share of every
    addition is its share of that day's moved energy, so each building-day keeps its own energy."""
    peak, target = _hours_mask(scenario.peak_hours), _hours_mask(scenario.target_hours)
    if not target.any():
        raise ValueError("The pre-cooling window is empty")
    if (peak & target).any():
        raise ValueError("The pre-cooling window overlaps the peak window")
    if not (0 <= scenario.shift_share <= 1 and 0 <= scenario.trim <= 1 and scenario.shift_loss >= 0):
        raise ValueError("Shift share and trim must be between 0 and 1, and the loss non-negative")
    taking_part = set(load["building"].unique() if taking_part is None else taking_part)

    days = pd.DatetimeIndex(np.unique(load["timestamp"].dt.normalize()))
    campus = np.zeros((len(days), SLOTS_PER_DAY))
    parts = []
    for building, rows in load.groupby("building", sort=False):
        grid, day, slot = _grid(rows, building)
        on = days.get_indexer(pd.DatetimeIndex(np.unique(rows["timestamp"].dt.normalize())))
        base, moved = np.nan, np.zeros_like(grid)
        after = grid.copy()
        if building in taking_part:
            ts = rows["timestamp"]
            base = base_load(rows["kW"], ts.dt.hour + ts.dt.minute / 60)
            after = np.minimum(grid, base) + np.clip(grid - base, 0, None) * (1 - scenario.trim)
            moved[:, peak] = np.clip(after[:, peak] - base, 0, None) * scenario.shift_share
            after -= moved
        campus[on] += after
        owed = np.zeros(len(days))
        owed[on] = moved.sum(axis=1) * (1 + scenario.shift_loss)
        parts.append((rows, base, after, day, slot, on, owed))

    owed_total = np.sum([owed for *_, owed in parts], axis=0)
    added = fill_valleys(campus[:, target], owed_total)
    frames = []
    for rows, base, after, day, slot, on, owed in parts:
        share = np.divide(owed, owed_total, out=np.zeros_like(owed), where=owed_total > 0)
        after[:, target] += added[on] * share[on, None]
        frames.append(rows.assign(base_kW=base, kW_after=after[day, slot]))
    return pd.concat(frames)


def in_window(timestamps: pd.Series, hours: tuple[int, int] = PEAK_HOURS) -> pd.Series:
    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    return (hour >= hours[0]) & (hour < hours[1])


def cost(timestamps: pd.Series, building: pd.Series, kw: pd.Series, tariff: Tariff,
         peak_hours: tuple[int, int] = PEAK_HOURS) -> pd.DataFrame:
    """Energy and demand charges per building."""
    peak = in_window(timestamps, peak_hours)
    kwh = kw / 4
    frame = pd.DataFrame({"building": building, "month": timestamps.dt.to_period("M"), "kW": kw,
                          "energy": np.where(peak, tariff.peak, tariff.off_peak) * kwh})
    monthly_max = frame.groupby(["building", "month"])["kW"].max()
    return pd.DataFrame({"energy": frame.groupby("building")["energy"].sum(),
                         "demand": monthly_max.groupby("building").sum() * tariff.demand})

