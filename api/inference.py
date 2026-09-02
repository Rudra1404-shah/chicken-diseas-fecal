"""The inference engine, deliberately separated from the HTTP layer.

Keeping this out of `main.py` means the routes can be tested against a stub
predictor without loading TensorFlow, and the engine can be reused by a batch
job or a CLI without dragging FastAPI along.

Preprocessing is imported from `src.data`, never reimplemented. Train/serve skew
from a subtly different resize is one of the most common ways a model that
scored 0.99 offline underperforms in production.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {
    "model_name": "unknown",
    "img_size": 224,
    "threshold": 0.5,
    "classes": {"0": "Healthy", "1": "Diseased"},
    "positive_class": "Diseased",
}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is available."""


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


@dataclass
class Prediction:
    label: str
    label_index: int
    probability_diseased: float
    threshold: float
    confidence: float
    inference_ms: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "label_index": self.label_index,
            "probability_diseased": round(self.probability_diseased, 6),
            "threshold": round(self.threshold, 6),
            "confidence": round(self.confidence, 6),
            "inference_ms": round(self.inference_ms, 2),
        }


class ChickenDiseaseClassifier:
    """Loads the exported .keras model and its metadata, then serves predictions."""

    def __init__(self, model_path: str | pathlib.Path, metadata_path: str | pathlib.Path):
        self.model_path = pathlib.Path(model_path)
        self.metadata_path = pathlib.Path(metadata_path)
        self.model = None
        self.metadata = dict(DEFAULT_METADATA)
        self.load_error: str | None = None

    # --- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load metadata first so /metadata works even if the weights are missing."""
        if self.metadata_path.exists():
            try:
                self.metadata.update(json.loads(self.metadata_path.read_text()))
            except json.JSONDecodeError as exc:
                logger.warning("metadata.json is not valid JSON: %s", exc)

        if not self.model_path.exists():
            self.load_error = (
                f"Model file not found at {self.model_path}. Train one with "
                f"`python -m src.train`, or copy chicken_model.keras into artifacts/."
            )
            logger.error(self.load_error)
            return

        try:
            from tensorflow import keras

            start = time.time()
            self.model = keras.models.load_model(self.model_path)
            logger.info("model loaded in %.1fs", time.time() - start)
            self._warmup()
        except Exception as exc:  # noqa: BLE001 — surface any load failure via /health
            self.load_error = f"Failed to load model: {exc}"
            logger.exception("model load failed")

    def _warmup(self) -> None:
        """One dummy pass so the first real request isn't paying graph-build cost.

        Without this the first user-facing prediction takes seconds instead of
        milliseconds, which looks like a broken service.
        """
        size = int(self.metadata.get("img_size", 224))
        dummy = np.zeros((1, size, size, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        logger.info("warmup complete")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def img_size(self) -> int:
        return int(self.metadata.get("img_size", 224))

    # --- prediction --------------------------------------------------------
    def preprocess(self, raw: bytes):
        from src.preprocessing import decode_image_bytes

        try:
            return decode_image_bytes(raw, self.img_size)
        except Exception as exc:  # noqa: BLE001
            raise InvalidImageError(
                "Could not decode that file as an image. Send a JPEG, PNG or BMP."
            ) from exc

    def predict(self, raw: bytes, threshold: float | None = None) -> Prediction:
        if not self.is_ready:
            raise ModelNotLoadedError(self.load_error or "Model is not loaded.")

        thr = self.threshold if threshold is None else float(threshold)
        batch = self.preprocess(raw)

        start = time.perf_counter()
        probability = float(self.model.predict(batch, verbose=0)[0][0])
        elapsed_ms = (time.perf_counter() - start) * 1000

        index = int(probability >= thr)
        return Prediction(
            label=self.metadata["classes"][str(index)],
            label_index=index,
            probability_diseased=probability,
            threshold=thr,
            # Distance from the threshold, normalised — how far from a coin flip
            # this call was. A score of 0.45 against a threshold of 0.42 is a
            # near-miss and the farmer should know that.
            confidence=abs(probability - thr) / max(thr, 1.0 - thr),
            inference_ms=elapsed_ms,
        )
