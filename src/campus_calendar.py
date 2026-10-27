"""Academic calendar: dates the facilities team knows in advance. Edit to match the college."""
import numpy as np
import pandas as pd

# Inclusive date ranges when classes stop and most students leave campus.
BREAKS = [
    ("2026-01-01", "2026-01-03"),  # tail of the winter break
    ("2026-05-16", "2026-07-19"),  # summer vacation
    ("2026-12-19", "2026-12-31"),  # winter break
]


def is_break(timestamps: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """True for every timestamp that falls inside an academic break."""
    days = pd.DatetimeIndex(timestamps).normalize()
    result = np.zeros(len(days), dtype=bool)
    for start, end in BREAKS:
        result |= (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    return result
