"""Forecasting model registry: one JSON file recording every model the backtests and the app may use.

A record says what the model is (baseline, local checkpoint, hosted endpoint, provider job), where
it lives, what data and configuration produced it and when its training data ends. ``status`` is
``verified`` only after the checkpoint or endpoint has been checked; provider jobs that have not
finished stay ``pending`` and are never used for forecasting.
"""
import fcntl
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import provenance

REGISTRY_PATH = Path("models/forecast_registry.json")
KINDS = {"baseline", "local_checkpoint", "hosted_endpoint", "provider_job"}
STATUSES = {"pending", "verified", "failed"}


@dataclass
class ModelRecord:
    name: str
    version: str
    kind: str
    location: str  # checkpoint dir, Hugging Face id, endpoint URL or provider job id
    status: str = "pending"
    base_model: str | None = None
    provider: str | None = None
    dataset_fingerprint: str | None = None
    training_cutoff: str | None = None
    config_hash: str | None = None
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    created_at: str = field(default_factory=provenance.now_utc)
    notes: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")


def load(path: Path | None = None) -> dict[str, ModelRecord]:
    raw = provenance.read_json(path or REGISTRY_PATH) or {}
    return {name: ModelRecord(**record) for name, record in raw.get("models", {}).items()}


def register(record: ModelRecord, path: Path | None = None) -> ModelRecord:
    """Add or replace a record (keyed by name). The file is locked while it is rewritten."""
    path = Path(path or REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = load(path)
        records[record.name] = record
        provenance.write_json(path, {"registry_version": 1, "models": {n: asdict(r) for n, r in records.items()}})
    return record


def get(name: str, path: Path | None = None) -> ModelRecord | None:
    return load(path).get(name)


def usable(record: ModelRecord | None, dataset_fingerprint: str | None = None) -> tuple[bool, str]:
    """Whether a record may serve forecasts, and why not."""
    if record is None:
        return False, "not registered"
    if record.status != "verified":
        return False, f"status is {record.status}"
    if record.kind == "local_checkpoint" and not Path(record.location).exists():
        return False, f"checkpoint {record.location} is missing"
    if dataset_fingerprint and record.dataset_fingerprint and record.kind == "local_checkpoint" \
            and record.dataset_fingerprint != dataset_fingerprint:
        return True, "trained on a different dataset version (usable, but re-benchmark before trusting its scores)"
    return True, "ok"
