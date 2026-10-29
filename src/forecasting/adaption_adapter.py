"""Adaption Labs integration for Chronos-2 fine-tuning.

VERIFIED (from the published ``adaption`` Python SDK, v0.16.0, inspected 2026-10-29):
  * Client ``adaption.Adaption``; auth by bearer token from ``ADAPTION_API_KEY``; base URL
    ``https://api.prod.adaptionlabs.ai`` unless ``ADAPTION_BASE_URL`` is set.
  * ``client.training_models.list()`` -> ``.models[]`` with ``id, display_name, context_length,
    methods, training_types`` (GET /api/v1/training-models).
  * ``client.datasets.upload_file(path, name=..., processing_mode=...)`` -> ``dataset_id``
    (.csv/.json/.jsonl/.parquet); ``client.datasets.get_status(dataset_id)``.
  * ``client.autoscientist.create(dataset_id=..., model=..., idempotency_key=...)`` -> run with
    ``id, status (pending|running|succeeded|failed|cancelled), error, download_available``;
    ``client.autoscientist.get(id)``; ``client.autoscientist.download(id)`` (gzip artifact).

NOT VERIFIED (no credentials, and nothing public describes it):
  * That the account lists a Chronos-2 model at all. The published SDK documents LLM training only
    (``training_method`` instruction/alignment, prompt/completion columns, token context length,
    win-rate optimisation). It has no time-series dataset format, horizon, covariate or
    forecasting-inference fields.
  * The time-series dataset layout Adaption would accept, the training objective it would use, and
    whether the artifact is an exportable Chronos-2 checkpoint or a hosted endpoint.

Therefore this adapter:
  1. refuses to upload or train unless ``training_models.list()`` returns a Chronos model id
     (``verify_capability``);
  2. never uploads data or starts paid training unless both ``ADAPTION_ALLOW_UPLOAD=1`` /
     ``ADAPTION_ALLOW_PAID_TRAINING=1`` are set *and* the caller passes ``confirm=True``;
  3. sends no LLM-specific fields (column mappings) it cannot justify for time series;
  4. persists every job id with the dataset fingerprint and configuration, and only registers a
     model as ``verified`` after the downloaded artifact passes ``verify_checkpoint``.
Secrets are read from the environment only and are never logged.
"""
import json
import logging
import os
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import provenance
from src.forecasting import registry
from src.forecasting.chronos_adapter import verify_checkpoint
from src.forecasting.interfaces import ForecasterUnavailable

log = logging.getLogger(__name__)
JOBS_DIR = Path("models/adaption_jobs")
TERMINAL = {"succeeded", "failed", "cancelled"}


class AdaptionConfigError(RuntimeError):
    """Missing credentials, SDK or explicit permission."""


class AdaptionCapabilityError(ForecasterUnavailable):
    """The provider account does not expose the needed Chronos-2 capability."""


@dataclass(frozen=True)
class AdaptionSettings:
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    chronos_model_id: str | None = None
    allow_upload: bool = False
    allow_paid_training: bool = False

    @classmethod
    def from_env(cls) -> "AdaptionSettings":
        return cls(api_key=os.environ.get("ADAPTION_API_KEY") or None,
                   base_url=os.environ.get("ADAPTION_BASE_URL") or None,
                   chronos_model_id=os.environ.get("ADAPTION_CHRONOS_MODEL_ID") or None,
                   allow_upload=os.environ.get("ADAPTION_ALLOW_UPLOAD") == "1",
                   allow_paid_training=os.environ.get("ADAPTION_ALLOW_PAID_TRAINING") == "1")


@dataclass
class JobRecord:
    job_id: str
    dataset_id: str
    model_id: str
    dataset_fingerprint: str
    config: dict
    status: str = "pending"
    error: str | None = None
    artifact_path: str | None = None
    submitted_at: str = field(default_factory=provenance.now_utc)
    updated_at: str = field(default_factory=provenance.now_utc)


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination.resolve())) or member.issym() or member.islnk():
                raise AdaptionConfigError(f"refusing unsafe path in artifact: {member.name}")
        tar.extractall(destination, filter="data")


class AdaptionChronosProvider:
    def __init__(self, settings: AdaptionSettings | None = None, client=None, jobs_dir: Path = JOBS_DIR,
                 registry_path: Path | None = None):
        self.settings = settings or AdaptionSettings.from_env()
        self._client = client
        self.jobs_dir, self.registry_path = Path(jobs_dir), registry_path

    # -- configuration -------------------------------------------------------------------------
    def check_configuration(self) -> list[str]:
        """Problems that prevent any live call (never includes secret values)."""
        problems = []
        if self._client is None:
            if not self.settings.api_key:
                problems.append("ADAPTION_API_KEY is not set")
            try:
                import adaption  # noqa: F401
            except ImportError:
                problems.append("the adaption SDK is not installed (pip install adaption)")
        return problems

    @property
    def client(self):
        if self._client is None:
            problems = self.check_configuration()
            if problems:
                raise AdaptionConfigError("; ".join(problems))
            import adaption

            self._client = adaption.Adaption(api_key=self.settings.api_key, base_url=self.settings.base_url)
        return self._client

    # -- capability ----------------------------------------------------------------------------
    def chronos_models(self) -> list[dict]:
        """Training models whose id or name mentions Chronos (GET /api/v1/training-models)."""
        response = self.client.training_models.list()
        return [{"id": m.id, "display_name": m.display_name, "methods": list(m.methods),
                 "training_types": list(m.training_types), "context_length": m.context_length}
                for m in response.models if "chronos" in f"{m.id} {m.display_name}".lower()]

    def verify_capability(self) -> dict:
        models = self.chronos_models()
        wanted = self.settings.chronos_model_id
        match = next((m for m in models if m["id"] == wanted), None) if wanted else (models[0] if len(models) == 1 else None)
        if match is None:
            listed = ", ".join(m["id"] for m in models) or "none"
            raise AdaptionCapabilityError(
                f"the Adaption account lists no usable Chronos-2 training model (requested {wanted!r}; Chronos models listed: {listed}). "
                "Set ADAPTION_CHRONOS_MODEL_ID to one of the listed ids.")
        return match

    # -- data and jobs -------------------------------------------------------------------------
    def upload_dataset(self, path: Path, name: str, confirm: bool = False) -> str:
        """Upload a prepared time-series file. Private data leaves the machine only with
        ``ADAPTION_ALLOW_UPLOAD=1`` and ``confirm=True``."""
        if not (self.settings.allow_upload and confirm):
            raise AdaptionConfigError("dataset upload needs ADAPTION_ALLOW_UPLOAD=1 and confirm=True")
        self.verify_capability()
        response = self.client.datasets.upload_file(Path(path), name=name)
        log.info("uploaded %s to Adaption as dataset %s", Path(path).name, response.dataset_id)
        return response.dataset_id

    def submit(self, dataset_id: str, dataset_fingerprint: str, config: dict, confirm: bool = False) -> JobRecord:
        """Start a (paid) training run and persist its job id."""
        if not (self.settings.allow_paid_training and confirm):
            raise AdaptionConfigError("training needs ADAPTION_ALLOW_PAID_TRAINING=1 and confirm=True")
        model = self.verify_capability()
        key = provenance.config_hash({"dataset": dataset_id, "fingerprint": dataset_fingerprint, "config": config})
        run = self.client.autoscientist.create(dataset_id=dataset_id, model=model["id"], idempotency_key=key)
        job = JobRecord(job_id=run.id, dataset_id=dataset_id, model_id=run.model or model["id"],
                        dataset_fingerprint=dataset_fingerprint, config=config, status=run.status)
        self._save(job)
        registry.register(registry.ModelRecord(
            name=f"chronos2_adaption_{run.id}", version=run.id, kind="provider_job", location=run.id, status="pending",
            base_model=job.model_id, provider="adaption", dataset_fingerprint=dataset_fingerprint,
            training_cutoff=config.get("train", [None, None])[1], config_hash=provenance.config_hash(config), config=config,
            notes="submitted; not usable until the artifact is downloaded and verified"), self.registry_path)
        return job

    def status(self, job_id: str) -> JobRecord:
        job = self._load(job_id)
        run = self.client.autoscientist.get(job_id)
        job.status, job.error, job.updated_at = run.status, run.error, provenance.now_utc()
        self._save(job)
        if run.status == "failed":
            log.error("Adaption job %s failed: %s", job_id, run.error)
        return job

    def retrieve(self, job_id: str, destination: Path = Path("models/checkpoints")) -> registry.ModelRecord:
        """Download a finished job's artifact, verify it is a Chronos-2 checkpoint, register it."""
        job = self.status(job_id)
        if job.status != "succeeded":
            raise AdaptionConfigError(f"job {job_id} is {job.status}, not succeeded")
        run = self.client.autoscientist.get(job_id)
        if not run.download_available:
            raise AdaptionCapabilityError(f"job {job_id} has no downloadable artifact (hosted-only artifacts are not supported yet)")
        target = Path(destination) / f"adaption_{job_id}"
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "artifact.tar.gz"
        self.client.autoscientist.download(job_id).write_to_file(archive)
        _safe_extract(archive, target)
        checkpoint = next((p.parent for p in target.rglob("config.json")), None) or next(
            (p.parent for p in target.rglob("adapter_config.json")), target)
        info = verify_checkpoint(checkpoint)  # raises if the artifact is not Chronos-2
        job.artifact_path, job.status = str(checkpoint), "succeeded"
        self._save(job)
        return registry.register(registry.ModelRecord(
            name=f"chronos2_adaption_{job_id}", version=job_id, kind="local_checkpoint", location=str(checkpoint),
            status="verified", base_model=job.model_id, provider="adaption", dataset_fingerprint=job.dataset_fingerprint,
            training_cutoff=job.config.get("train", [None, None])[1], config_hash=provenance.config_hash(job.config),
            config=job.config, environment=provenance.environment(), notes=json.dumps(info)), self.registry_path)

    def _save(self, job: JobRecord) -> None:
        provenance.write_json(self.jobs_dir / f"{job.job_id}.json", asdict(job))

    def _load(self, job_id: str) -> JobRecord:
        raw = provenance.read_json(self.jobs_dir / f"{job_id}.json")
        if raw is None:
            raise AdaptionConfigError(f"no local record of job {job_id}")
        return JobRecord(**raw)
