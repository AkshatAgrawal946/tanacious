"""Contract tests for the Adaption Labs adapter against a mocked client (no network, no cost).

The mock mirrors the response fields of the published ``adaption`` SDK v0.16.0; whether the live
service lists a Chronos-2 model is NOT verified by these tests.
"""
import io
import json
import logging
import tarfile
import types

import pytest

from src.forecasting import registry
from src.forecasting.adaption_adapter import (AdaptionCapabilityError, AdaptionChronosProvider, AdaptionConfigError,
                                              AdaptionSettings)

SECRET = "sk-test-should-never-appear"


def _model(id_, name):
    return types.SimpleNamespace(id=id_, display_name=name, methods=["sft"], training_types=["lora"], context_length=8192)


def _tarball(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeClient:
    def __init__(self, models, artifact: bytes | None = None, final_status="succeeded"):
        self.created, self.uploaded = [], []
        self.run = types.SimpleNamespace(id="run_123", model=None, status="pending", error=None, download_available=False)
        client = self

        class Download:
            def write_to_file(self, path):
                open(path, "wb").write(artifact)

        self.training_models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(models=models))
        self.datasets = types.SimpleNamespace(upload_file=lambda path, name=None: (
            client.uploaded.append((path, name)), types.SimpleNamespace(dataset_id="ds_1", status="pending"))[1])

        def create(**kwargs):
            client.created.append(kwargs)
            client.run.model = kwargs["model"]
            return client.run

        def get(job_id):
            client.run.status = final_status
            client.run.download_available = artifact is not None and final_status == "succeeded"
            client.run.error = "out of memory" if final_status == "failed" else None
            return client.run

        self.autoscientist = types.SimpleNamespace(create=create, get=get, download=lambda job_id: Download())


CHRONOS_CONFIG = json.dumps({"architectures": ["Chronos2Model"], "chronos_config": {"context_length": 8192}})


def _provider(tmp_path, client, **settings):
    defaults = dict(api_key=SECRET, chronos_model_id="chronos-2", allow_upload=True, allow_paid_training=True)
    return AdaptionChronosProvider(AdaptionSettings(**{**defaults, **settings}), client=client,
                                   jobs_dir=tmp_path / "jobs", registry_path=tmp_path / "registry.json")


def test_missing_credentials_are_reported_without_secrets(monkeypatch):
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    provider = AdaptionChronosProvider(AdaptionSettings.from_env())
    assert "ADAPTION_API_KEY is not set" in provider.check_configuration()
    with pytest.raises(AdaptionConfigError):
        provider.client  # noqa: B018


def test_settings_repr_hides_the_key():
    assert SECRET not in repr(AdaptionSettings(api_key=SECRET))


def test_capability_check_refuses_accounts_without_chronos(tmp_path):
    provider = _provider(tmp_path, FakeClient([_model("llama-3-8b", "Llama 3 8B")]))
    with pytest.raises(AdaptionCapabilityError, match="no usable Chronos-2"):
        provider.verify_capability()
    with pytest.raises(AdaptionCapabilityError):
        provider.upload_dataset(tmp_path / "train.jsonl", "x", confirm=True)


def test_upload_and_training_need_explicit_opt_in(tmp_path):
    client = FakeClient([_model("chronos-2", "Chronos-2")])
    with pytest.raises(AdaptionConfigError, match="ADAPTION_ALLOW_UPLOAD"):
        _provider(tmp_path, client).upload_dataset(tmp_path / "t.jsonl", "x", confirm=False)
    with pytest.raises(AdaptionConfigError, match="ADAPTION_ALLOW_PAID_TRAINING"):
        _provider(tmp_path, client, allow_paid_training=False).submit("ds_1", "fp", {}, confirm=True)
    assert client.created == [] and client.uploaded == []


def test_submit_persists_job_and_registers_pending_record(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    client = FakeClient([_model("chronos-2", "Chronos-2")])
    provider = _provider(tmp_path, client)
    job = provider.submit("ds_1", "fp123", {"train": ["2026-01-01", "2026-08-01"]}, confirm=True)
    assert job.job_id == "run_123" and client.created[0]["model"] == "chronos-2"
    assert "idempotency_key" in client.created[0]
    stored = json.loads((tmp_path / "jobs" / "run_123.json").read_text())
    assert stored["dataset_fingerprint"] == "fp123"
    record = registry.get("chronos2_adaption_run_123", tmp_path / "registry.json")
    assert record.status == "pending" and registry.usable(record) == (False, "status is pending")
    assert SECRET not in caplog.text and SECRET not in (tmp_path / "registry.json").read_text()


def test_failed_job_is_reported_and_not_registered_as_usable(tmp_path):
    provider = _provider(tmp_path, FakeClient([_model("chronos-2", "Chronos-2")], final_status="failed"))
    provider.submit("ds_1", "fp", {}, confirm=True)
    assert provider.status("run_123").error == "out of memory"
    with pytest.raises(AdaptionConfigError, match="failed"):
        provider.retrieve("run_123", tmp_path / "ckpt")


def test_retrieve_verifies_and_registers_checkpoint(tmp_path):
    artifact = _tarball({"model/config.json": CHRONOS_CONFIG, "model/model.safetensors": "weights"})
    provider = _provider(tmp_path, FakeClient([_model("chronos-2", "Chronos-2")], artifact=artifact))
    provider.submit("ds_1", "fp", {"train": ["2026-01-01", "2026-08-01"]}, confirm=True)
    record = provider.retrieve("run_123", tmp_path / "ckpt")
    assert record.status == "verified" and record.kind == "local_checkpoint"
    assert record.training_cutoff == "2026-08-01" and record.location.endswith("model")


def test_retrieve_rejects_non_chronos_artifacts(tmp_path):
    artifact = _tarball({"config.json": json.dumps({"architectures": ["LlamaForCausalLM"]})})
    provider = _provider(tmp_path, FakeClient([_model("chronos-2", "Chronos-2")], artifact=artifact))
    provider.submit("ds_1", "fp", {}, confirm=True)
    with pytest.raises(Exception, match="not a Chronos-2 checkpoint"):
        provider.retrieve("run_123", tmp_path / "ckpt")
    assert registry.get("chronos2_adaption_run_123", tmp_path / "registry.json").status == "pending"


def test_retrieve_rejects_path_traversal(tmp_path):
    artifact = _tarball({"../evil.json": "{}"})
    provider = _provider(tmp_path, FakeClient([_model("chronos-2", "Chronos-2")], artifact=artifact))
    provider.submit("ds_1", "fp", {}, confirm=True)
    with pytest.raises(AdaptionConfigError, match="unsafe path"):
        provider.retrieve("run_123", tmp_path / "ckpt")
    assert not (tmp_path / "evil.json").exists()


def test_settings_read_dotenv_without_overriding_shell(tmp_path, monkeypatch):
    for name in ("ADAPTION_API_KEY", "ADAPTION_CHRONOS_MODEL_ID", "ADAPTION_ALLOW_UPLOAD"):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nADAPTION_API_KEY='from-file'\nexport ADAPTION_CHRONOS_MODEL_ID=chronos-2\n")
    monkeypatch.setenv("ADAPTION_ALLOW_UPLOAD", "0")
    settings = AdaptionSettings.from_env(dotenv)
    assert settings.api_key == "from-file" and settings.chronos_model_id == "chronos-2"
    monkeypatch.setenv("ADAPTION_API_KEY", "from-shell")
    assert AdaptionSettings.from_env(dotenv).api_key == "from-shell"


def test_placeholder_key_counts_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTION_API_KEY", "your-adaption-api-key")
    assert AdaptionSettings.from_env(None).api_key is None
