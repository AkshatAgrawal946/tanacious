"""Chronos-2 fine-tuning commands.

    python -m src.forecasting.finetune prepare                  # build + validate training data (no network)
    python -m src.forecasting.finetune local --steps 300         # LoRA fine-tune locally, register checkpoint
    python -m src.forecasting.finetune adaption-check            # credentials + Chronos capability (read-only call)
    python -m src.forecasting.finetune adaption-submit --confirm # upload + paid training (needs env opt-ins)
    python -m src.forecasting.finetune adaption-status JOB_ID
    python -m src.forecasting.finetune adaption-retrieve JOB_ID

Training uses the training split only, validation (checkpoint selection) the validation split only;
the test split is never touched here.
"""
import argparse
import logging
import sys
from pathlib import Path

from src import provenance
from src.data import load_data
from src.forecasting import registry
from src.forecasting.adaption_adapter import AdaptionChronosProvider, AdaptionConfigError
from src.forecasting.config import ForecastExperimentConfig
from src.forecasting.data_prep import (building_series, chronos_training_inputs, chronos_validation_inputs, export_jsonl,
                                       known_covariate_names, to_chronos, validate_training_inputs)
from src.forecasting.interfaces import ForecasterUnavailable
from src.ml_model import campus_temperature

log = logging.getLogger(__name__)
TRAINING_DIR = Path("data/processed/forecast_training")
LOCAL_NAME = "chronos2_finetuned_local"


def prepare(config: ForecastExperimentConfig, output_dir: Path = TRAINING_DIR) -> dict:
    """Build, validate and export the training and validation series; raise on any problem."""
    interval, bills = load_data()
    series = {b: building_series(interval, b) for b in config.buildings}
    temperature = campus_temperature(interval)
    train_start, train_end = config.split("train")
    val_start, val_end = config.split("validation")
    train = chronos_training_inputs(series, train_start, train_end, config.horizon_steps, config.covariate_scenario, temperature)
    validation = chronos_validation_inputs(series, val_start, val_end, config.context_steps, config.horizon_steps,
                                           config.origin_stride_steps, config.covariate_scenario, temperature,
                                           config.max_missing_target_fraction)
    problems = validate_training_inputs(train, config.horizon_steps, split_end=train_end) + \
        validate_training_inputs(validation, config.horizon_steps, split_end=val_end)
    if problems:
        raise ValueError("training data is not valid: " + "; ".join(problems[:10]))
    known = known_covariate_names(config.covariate_scenario)
    manifest = {"dataset_fingerprint": provenance.dataset_fingerprint(interval, bills), "config": config.to_dict(),
                "config_hash": provenance.config_hash(config.to_dict()), "known_covariates": known,
                "train_series": len(train), "train_steps": int(sum(len(i["target"]) for i in train)),
                "validation_series": len(validation), "training_cutoff": str(train_end),
                "files": ["train.jsonl", "validation.jsonl"], "created_at": provenance.now_utc()}
    export_jsonl(train, output_dir / "train.jsonl", known)
    export_jsonl(validation, output_dir / "validation.jsonl", known)
    provenance.write_json(output_dir / "manifest.json", manifest)
    return {"train": train, "validation": validation, "manifest": manifest}


def local(config: ForecastExperimentConfig, steps: int, learning_rate: float, batch_size: int, mode: str,
          device: str) -> registry.ModelRecord:
    from src.forecasting.chronos_adapter import finetune_local, verify_checkpoint

    data = prepare(config)
    manifest = data["manifest"]
    output = Path("models/checkpoints") / f"{LOCAL_NAME}_{manifest['config_hash']}"
    checkpoint = finetune_local(to_chronos(data["train"]), to_chronos(data["validation"]), manifest["known_covariates"],
                                config.horizon_steps, config.context_steps, output, base_model=config.chronos_model_id,
                                mode=mode, num_steps=steps, learning_rate=learning_rate, batch_size=batch_size,
                                device=device, seed=config.seed)
    hyper = {"mode": mode, "steps": steps, "learning_rate": learning_rate, "batch_size": batch_size, "device": device}
    return registry.register(registry.ModelRecord(
        name=LOCAL_NAME, version=f"{manifest['config_hash']}-{provenance.config_hash(hyper)}", kind="local_checkpoint",
        location=str(checkpoint), status="verified", base_model=config.chronos_model_id, provider="local",
        dataset_fingerprint=manifest["dataset_fingerprint"], training_cutoff=manifest["training_cutoff"],
        config_hash=manifest["config_hash"], config={**config.to_dict(), "finetune": hyper},
        environment=provenance.environment(), notes=str(verify_checkpoint(checkpoint))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chronos-2 fine-tuning")
    parser.add_argument("command", choices=["prepare", "local", "adaption-check", "adaption-submit",
                                            "adaption-status", "adaption-retrieve"])
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--config", default="configs/forecast_benchmark.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirm", action="store_true", help="explicitly approve upload and paid training")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ForecastExperimentConfig.from_json(args.config) if Path(args.config).exists() else ForecastExperimentConfig()
    try:
        if args.command == "prepare":
            manifest = prepare(config)["manifest"]
            print(f"prepared {manifest['train_series']} training series ({manifest['train_steps']:,} steps) and "
                  f"{manifest['validation_series']} validation windows in {TRAINING_DIR}")
        elif args.command == "local":
            record = local(config, args.steps, args.learning_rate, args.batch_size, args.mode, args.device)
            print(f"registered {record.name} ({record.version}) at {record.location}")
        else:
            provider = AdaptionChronosProvider()
            if args.command == "adaption-check":
                problems = provider.check_configuration()
                if problems:
                    print("Adaption not configured: " + "; ".join(problems))
                    return 2
                print(f"Adaption Chronos-2 model available: {provider.verify_capability()}")
            elif args.command == "adaption-submit":
                manifest = prepare(config)["manifest"]
                dataset_id = provider.upload_dataset(TRAINING_DIR / "train.jsonl", "tanacious-chronos2-train", confirm=args.confirm)
                job = provider.submit(dataset_id, manifest["dataset_fingerprint"], config.to_dict(), confirm=args.confirm)
                print(f"submitted Adaption job {job.job_id} (status {job.status}); poll with adaption-status {job.job_id}")
            elif args.command == "adaption-status":
                job = provider.status(args.job_id)
                print(f"{job.job_id}: {job.status}" + (f" ({job.error})" if job.error else ""))
            elif args.command == "adaption-retrieve":
                record = provider.retrieve(args.job_id)
                print(f"registered {record.name} at {record.location}")
    except (AdaptionConfigError, ForecasterUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
