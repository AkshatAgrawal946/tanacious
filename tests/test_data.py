import pandas as pd

from src.data import validate_inputs


def test_validator_reports_required_columns():
    issues = validate_inputs(pd.DataFrame(), pd.DataFrame())
    assert len(issues) == 2
    assert "interval data is missing columns" in issues[0]
