"""Chronos-2 forecasting adapter (local inference with the open-weight ``chronos-forecasting`` package).

Nothing heavy is imported at module import: torch and the model weights load on first use. Every
failure to load or predict becomes ``ForecasterUnavailable`` so callers can fall back to a baseline
and record why. The model is used zero-shot (no fine-tuning).
"""
import concurrent.futures
import logging
import time
from typing import Sequence

from src.forecasting.interfaces import ForecasterUnavailable, ForecastRequest, ForecastResult, ForecastTimeout, validate_result

log = logging.getLogger(__name__)
ZERO_SHOT_ID = "amazon/chronos-2"


def select_device(requested: str = "cpu") -> str:
    """'cpu', 'cuda', 'mps' or 'auto' (CUDA if present, else CPU). CPU is the reproducible default."""
    try:
        import torch
    except ImportError as error:
        raise ForecasterUnavailable(f"torch is not installed: {error}") from error
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ForecasterUnavailable("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ForecasterUnavailable("MPS was requested but is not available")
    return requested


class ChronosForecaster:
    def __init__(self, model_id: str = ZERO_SHOT_ID, name: str = "chronos2_zero_shot",
                 version: str | None = None, device: str = "cpu", quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
                 timeout_s: float = 300.0, batch_size: int = 64, seed: int = 2026):
        self.model_id, self.name = model_id, name
        self.version = version or model_id
        self.device_request, self.timeout_s, self.batch_size, self.seed = device, timeout_s, batch_size, seed
        self.quantile_levels = sorted(set(quantile_levels) | {0.5})
        self._pipeline = None
        self.device: str | None = None

    def load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as error:
            raise ForecasterUnavailable(f"chronos-forecasting is not installed ({error}); "
                                        "pip install -r requirements-forecast.txt") from error
        self.device = select_device(self.device_request)
        torch.manual_seed(self.seed)
        try:
            start = time.perf_counter()
            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
            log.info("loaded %s on %s in %.1fs", self.model_id, self.device, time.perf_counter() - start)
        except Exception as error:  # network, missing weights, incompatible checkpoint
            raise ForecasterUnavailable(f"could not load Chronos-2 from {self.model_id}: {error}") from error
        return self._pipeline

    @staticmethod
    def _inputs(request: ForecastRequest) -> dict:
        item = {"target": request.history.to_numpy(dtype=float)}
        if request.past_covariates is not None:
            item["past_covariates"] = {c: request.past_covariates[c].to_numpy(dtype=float) for c in request.past_covariates}
        if request.future_covariates is not None:
            item["future_covariates"] = {c: request.future_covariates[c].to_numpy(dtype=float) for c in request.future_covariates}
        return item

    def _run(self, pipeline, requests: Sequence[ForecastRequest]):
        horizon = requests[0].horizon
        if any(r.horizon != horizon for r in requests):
            raise ValueError("all requests in one call must share the horizon")
        return pipeline.predict_quantiles([self._inputs(r) for r in requests], prediction_length=horizon,
                                          quantile_levels=self.quantile_levels, batch_size=self.batch_size)

    def predict(self, requests: Sequence[ForecastRequest]) -> list[ForecastResult]:
        if not requests:
            return []
        pipeline = self.load()
        start = time.perf_counter()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._run, pipeline, requests)
        try:
            quantiles, _ = future.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError as error:
            raise ForecastTimeout(f"{self.name} did not finish {len(requests)} forecasts within {self.timeout_s}s") from error
        except Exception as error:
            raise ForecasterUnavailable(f"{self.name} inference failed: {error}") from error
        finally:
            executor.shutdown(wait=False)
        latency = (time.perf_counter() - start) / len(requests)
        results = []
        for request, q in zip(requests, quantiles):
            values = q[0].detach().cpu().numpy()  # (horizon, n_quantiles) for the single target
            by_level = {level: values[:, i].astype(float) for i, level in enumerate(self.quantile_levels)}
            results.append(validate_result(request, ForecastResult(
                building=request.building, origin=request.origin, timestamps=request.forecast_index,
                mean=by_level[0.5].copy(), quantiles=by_level,
                model_name=self.name, model_version=self.version, latency_s=latency,
                metadata={"device": self.device, "model_id": self.model_id})))
        return results
