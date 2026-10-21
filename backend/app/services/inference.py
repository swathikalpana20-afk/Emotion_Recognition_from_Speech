"""
EMOTISENSE AI — Inference Service
Loads the trained model and runs predictions on audio input.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core.config import (
    ARTIFACTS_DIR,
    EMOTION_LABELS,
    FEATURE_DIM,
    MAX_TIME_STEPS,
    MODEL_CHECKPOINT,
    MODEL_VERSION,
    NUM_CLASSES,
    PREPROCESSING_CONFIG,
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
)
from app.ml.preprocessing import AudioPreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)

# Lazy import of torch-dependent modules
_model_module = None


def _get_model_module():
    global _model_module
    if _model_module is None:
        from app.ml import model as m
        _model_module = m
    return _model_module


class InferenceEngine:
    """
    Singleton-style inference engine.
    Loads model + preprocessing config once, then processes requests.
    """

    def __init__(self):
        self._model = None
        self._preprocessor: Optional[AudioPreprocessor] = None
        self._metadata: Dict = {}
        self._loaded: bool = False
        self._load_error: Optional[str] = None

    def load(self, checkpoint_path: Path = MODEL_CHECKPOINT) -> bool:
        """
        Attempt to load model and preprocessing config.
        Returns True if successful, False otherwise.
        """
        self._load_error = None
        self._loaded = False

        # 1. Load preprocessing config
        prep_cfg = PreprocessingConfig()  # fallback defaults
        if PREPROCESSING_CONFIG.exists():
            try:
                prep_cfg = PreprocessingConfig.load(PREPROCESSING_CONFIG)
                logger.info("Preprocessing config loaded from disk.")
            except Exception as exc:
                logger.warning(f"Could not load preprocessing config: {exc}. Using defaults.")

        self._preprocessor = AudioPreprocessor(prep_cfg)

        # 2. Load normalization stats
        stats_path = ARTIFACTS_DIR / "norm_stats.npz"
        if stats_path.exists():
            try:
                self._preprocessor.load_stats(stats_path)
                logger.info("Normalization stats loaded.")
            except Exception as exc:
                logger.warning(f"Could not load norm stats: {exc}")

        # 3. Load model checkpoint
        if not checkpoint_path.exists():
            msg = (
                f"Model checkpoint not found at {checkpoint_path}. "
                "Please train the model first using: "
                "python scripts/train.py --dataset ravdess"
            )
            logger.warning(msg)
            self._load_error = msg
            return False

        try:
            mm = _get_model_module()
            model, metadata = mm.load_checkpoint(
                checkpoint_path,
                model_type=self._metadata.get("model_type", "cnn_bilstm"),
                input_dim=prep_cfg.feature_dim,
                num_classes=NUM_CLASSES,
            )
            self._model = model
            self._metadata = metadata
            self._loaded = True
            logger.info(f"Model loaded successfully from {checkpoint_path}")
            return True
        except Exception as exc:
            self._load_error = f"Failed to load model: {exc}"
            logger.error(self._load_error)
            return False

    def predict(self, audio_bytes: bytes, filename: str = "upload.wav") -> Dict:
        """
        Run full inference pipeline on audio bytes.
        Returns structured prediction dict.
        """
        if not self._loaded:
            raise RuntimeError(
                self._load_error or "Model is not loaded. Train and load a model first."
            )

        # 1. Preprocess
        features, info = self._preprocessor.process_bytes(audio_bytes, filename)

        # 2. Silence warning
        silence_warning = info.get("is_silent", False)

        # 3. Tensor conversion
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch not installed.")

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, T, F]
        device = next(self._model.parameters()).device
        x = x.to(device)

        # 4. Forward pass
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # 5. Validate outputs
        assert len(probs) == NUM_CLASSES, f"Expected {NUM_CLASSES} probs, got {len(probs)}"
        if not np.all(np.isfinite(probs)):
            raise RuntimeError("Non-finite values in model output.")

        # 6. Build result
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        predicted_emotion = EMOTION_LABELS[predicted_idx]

        LOW_CONFIDENCE_THRESHOLD = 0.4
        low_confidence_warning = confidence < LOW_CONFIDENCE_THRESHOLD

        probabilities = [
            {"label": EMOTION_LABELS[i], "probability": float(probs[i])}
            for i in range(NUM_CLASSES)
        ]

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": self._metadata.get("model_version", MODEL_VERSION),
            "preprocessing_version": self._metadata.get(
                "preprocessing_version", PREPROCESSING_VERSION
            ),
            "low_confidence_warning": low_confidence_warning,
            "silence_warning": silence_warning,
            "mfcc_shape": list(features.shape),
            "audio_duration_seconds": info.get("duration_seconds", 0.0),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def metadata(self) -> Dict:
        return self._metadata


# ── Singleton instance ────────────────────────────────────────────
_engine = InferenceEngine()


def get_engine() -> InferenceEngine:
    """Return the global inference engine (FastAPI dependency)."""
    return _engine


def startup_load() -> None:
    """Called at FastAPI startup to try loading the model."""
    _engine.load()
