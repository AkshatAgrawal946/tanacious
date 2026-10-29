"""Reproducibility helpers: dataset fingerprints, environment records and JSON artifacts.

A fingerprint is a content hash, so two runs on identical data share it and any edit to the data
changes it. Model and evaluation artifacts store the fingerprint of the data they were built from;
consumers (the dashboard, the backtests) refuse artifacts whose fingerprint does not match.
"""
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
PACKAGES = ["numpy", "pandas", "scikit-learn", "streamlit"]


def frame_fingerprint(*frames: pd.DataFrame) -> str:
    """Order-sensitive content hash of one or more DataFrames (values, columns and dtypes)."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(",".join(map(str, frame.columns)).encode())
        digest.update(",".join(map(str, frame.dtypes)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


def dataset_fingerprint(interval: pd.DataFrame, bills: pd.DataFrame) -> str:
    """Fingerprint of the operational inputs, independent of row order."""
    interval = interval.sort_values(["building", "timestamp"]).reset_index(drop=True)
    bills = bills.assign(month=bills["month"].astype(str)).sort_values(["building", "month"]).reset_index(drop=True)
    return frame_fingerprint(interval[sorted(interval.columns)], bills[sorted(bills.columns)])


def file_fingerprint(*paths: Path) -> str:
    """Content hash of files on disk (missing files hash as absent)."""
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes() if path.exists() else b"<absent>")
    return digest.hexdigest()[:16]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "git_commit": git_commit()}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("ignoring unreadable JSON artifact %s", path)
        return None
