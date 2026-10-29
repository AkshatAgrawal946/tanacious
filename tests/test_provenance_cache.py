"""Dataset fingerprints, model metadata validation and dashboard cache keys."""
import ast
from pathlib import Path

import pytest

from src import ml_model, provenance
from src.generate_data import generate


@pytest.fixture(scope="module")
def campus(tmp_path_factory):
    interval, bills, _ = generate(tmp_path_factory.mktemp("raw"))
    return interval, bills


def test_fingerprint_is_order_independent_and_content_sensitive(campus):
    interval, bills = campus
    base = provenance.dataset_fingerprint(interval, bills)
    assert base == provenance.dataset_fingerprint(interval.sample(frac=1, random_state=0), bills.iloc[::-1])
    changed = interval.copy()
    changed.loc[changed.index[100], "kWh_consumed"] += 0.001
    assert provenance.dataset_fingerprint(changed, bills) != base
    assert provenance.dataset_fingerprint(interval, bills.assign(monthly_kWh=bills.monthly_kWh + 1)) != base


def test_file_fingerprint_tracks_content(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n")
    first = provenance.file_fingerprint(path)
    path.write_text("x\n2\n")
    assert provenance.file_fingerprint(path) != first


def test_saved_model_is_reused_only_for_matching_data_and_version(campus, tmp_path, monkeypatch):
    interval, bills = campus
    small = interval[interval.timestamp < "2026-02-15"]
    model = ml_model.train(small, bills)
    path = tmp_path / "rf.joblib"
    fingerprint = provenance.dataset_fingerprint(small, bills)
    ml_model.save(model, path, dataset_fingerprint=fingerprint)
    assert ml_model.load_compatible(fingerprint, path) is not None
    # Same features, different data: must not be reused.
    assert ml_model.load_compatible(provenance.dataset_fingerprint(interval, bills), path) is None
    # Same data, new model version: must not be reused.
    monkeypatch.setattr(ml_model, "MODEL_VERSION", "rf-reconstruction-999")
    assert ml_model.load_compatible(fingerprint, path) is None


def test_model_without_metadata_is_not_trusted(campus, tmp_path):
    import joblib

    interval, bills = campus
    path = tmp_path / "legacy.joblib"
    joblib.dump(ml_model.train(interval[interval.timestamp < "2026-02-01"], bills), path)
    assert ml_model.load_compatible("anything", path) is None


def test_training_refuses_ground_truth(campus):
    interval, bills = campus
    tagged = interval.copy()
    tagged.attrs["evaluation_only"] = True
    with pytest.raises(ValueError, match="ground truth"):
        ml_model.train(tagged, bills)


def test_app_cache_functions_are_keyed_by_fingerprints():
    """Streamlit leaves underscore-prefixed arguments out of the cache key; no cached function may
    take one, and the model/reconstruction caches must be keyed by data and model version."""
    tree = ast.parse(Path("app.py").read_text())
    cached = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, (ast.Attribute, ast.Call)) and "cache" in ast.unparse(d) for d in node.decorator_list):
            cached[node.name] = [a.arg for a in node.args.args]
    assert cached, "expected cached functions in app.py"
    for name, args in cached.items():
        assert not any(a.startswith("_") for a in args), f"{name} has an unhashed argument"
    assert {"files_signature", "dataset_fingerprint", "model_version"} <= set(cached["forest"])
    assert {"files_signature", "dataset_fingerprint", "model_version"} <= set(cached["reconstruction"])
    assert "files_signature" in cached["campus_data"]
    assert "dataset_fingerprint" in cached["tested_accuracy"]
