"""
EMOTISENSE AI — Audio Preprocessing Pipeline
Shared reusable pipeline for training and inference.
No train/inference mismatch: normalization stats are saved and reloaded.
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy deps
_librosa = None
_soundfile = None


def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


def _get_soundfile():
    global _soundfile
    if _soundfile is None:
        import soundfile as sf
        _soundfile = sf
    return _soundfile


class PreprocessingConfig:
    """Immutable snapshot of preprocessing hyperparameters."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_time_steps: int = 216,
        use_delta: bool = True,
        use_delta2: bool = True,
        silence_threshold_db: float = -60.0,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        version: str = "1.0.0",
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_time_steps = max_time_steps
        self.use_delta = use_delta
        self.use_delta2 = use_delta2
        self.silence_threshold_db = silence_threshold_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.version = version

    @property
    def feature_dim(self) -> int:
        dim = self.n_mfcc
        if self.use_delta:
            dim += self.n_mfcc
        if self.use_delta2:
            dim += self.n_mfcc
        return dim

    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "max_time_steps": self.max_time_steps,
            "use_delta": self.use_delta,
            "use_delta2": self.use_delta2,
            "silence_threshold_db": self.silence_threshold_db,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "version": self.version,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PreprocessingConfig":
        d = {k: v for k, v in d.items() if k != "feature_dim"}
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Preprocessing config saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "PreprocessingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Global default config (override at training time)
DEFAULT_CONFIG = PreprocessingConfig()


class AudioPreprocessor:
    """
    Converts raw audio bytes/path to normalized MFCC feature tensor.

    Usage:
        preprocessor = AudioPreprocessor(config)
        preprocessor.fit(list_of_audio_paths)   # compute normalization stats
        features = preprocessor.process_file(path)  # returns np.ndarray [T, F]
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────

    def process_bytes(self, audio_bytes: bytes, filename: str = "upload.wav") -> Tuple[np.ndarray, Dict]:
        """
        Process raw audio bytes → normalized feature array.
        Returns (features [T, F], info dict).
        """
        suffix = Path(filename).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            return self.process_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def process_file(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Full pipeline: load → resample → amplitude → silence check
        → MFCC → delta → pad/truncate → normalize.
        Returns (features [T, F], info dict).
        """
        librosa = _get_librosa()
        cfg = self.config

        # 1. Load audio
        try:
            y, sr = librosa.load(str(path), sr=cfg.sample_rate, mono=True, duration=cfg.max_duration)
        except Exception as exc:
            raise ValueError(f"Cannot decode audio file '{path.name}': {exc}") from exc

        duration = len(y) / cfg.sample_rate

        # 2. Duration validation
        if duration < cfg.min_duration:
            raise ValueError(
                f"Audio too short ({duration:.2f}s). Minimum is {cfg.min_duration}s."
            )

        # 3. Silence / near-silent check
        rms_db = _rms_db(y)
        is_silent = rms_db < cfg.silence_threshold_db

        # 4. Amplitude normalization (peak normalization, avoid div-by-zero)
        peak = np.abs(y).max()
        if peak > 1e-8:
            y = y / peak

        # 5. Extract MFCCs  [n_mfcc, T]
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

        # 6. Delta features
        features = mfcc
        if cfg.use_delta:
            delta = librosa.feature.delta(mfcc)
            features = np.concatenate([features, delta], axis=0)
        if cfg.use_delta2:
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([features, delta2], axis=0)

        # features shape: [feature_dim, T]
        assert features.shape[0] == cfg.feature_dim, (
            f"Feature dim mismatch: expected {cfg.feature_dim}, got {features.shape[0]}"
        )

        # 7. Transpose → [T, feature_dim]
        features = features.T  # [T, F]

        # 8. Pad / truncate along time axis
        features = _pad_or_truncate(features, cfg.max_time_steps)

        # 9. Validate finite values
        if not np.all(np.isfinite(features)):
            raise ValueError("Non-finite values detected in MFCC features.")

        # 10. Normalize using saved stats (inference) or pass raw (training fit)
        if self._mean is not None and self._std is not None:
            features = (features - self._mean) / (self._std + 1e-8)

        info = {
            "duration_seconds": float(duration),
            "rms_db": float(rms_db),
            "is_silent": bool(is_silent),
            "shape": list(features.shape),
            "feature_dim": cfg.feature_dim,
        }
        return features, info

    def fit(self, audio_paths: list[Path], max_samples: int = 5000) -> "AudioPreprocessor":
        """
        Compute per-feature-dimension mean/std from training data.
        Must be called only on training set, never on test set.
        """
        logger.info(f"Fitting normalization stats on {min(len(audio_paths), max_samples)} samples...")
        all_features = []
        for p in audio_paths[:max_samples]:
            try:
                feat, _ = self._extract_raw(p)
                all_features.append(feat)
            except Exception as exc:
                logger.warning(f"Skipping {p.name}: {exc}")

        if not all_features:
            raise RuntimeError("No valid audio files to fit normalization.")

        stacked = np.concatenate(all_features, axis=0)  # [N*T, F]
        self._mean = stacked.mean(axis=0, keepdims=True)  # [1, F]
        self._std = stacked.std(axis=0, keepdims=True)    # [1, F]
        logger.info(f"Normalization fit complete. mean shape: {self._mean.shape}")
        return self

    def save_stats(self, path: Path) -> None:
        """Save normalization statistics to disk."""
        if self._mean is None:
            raise RuntimeError("No normalization stats to save. Call .fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self._mean, std=self._std)
        logger.info(f"Normalization stats saved to {path}")

    def load_stats(self, path: Path) -> "AudioPreprocessor":
        """Load normalization statistics from disk."""
        data = np.load(str(path))
        self._mean = data["mean"]
        self._std = data["std"]
        logger.info(f"Normalization stats loaded from {path}")
        return self

    # ── Internal ─────────────────────────────────────────────────

    def _extract_raw(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Extract features WITHOUT applying normalization (used during fit)."""
        mean_backup, std_backup = self._mean, self._std
        self._mean, self._std = None, None
        try:
            return self.process_file(path)
        finally:
            self._mean, self._std = mean_backup, std_backup


# ── Helpers ────────────────────────────────────────────────────────

def _rms_db(y: np.ndarray) -> float:
    """Compute RMS energy in dB."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


def _pad_or_truncate(arr: np.ndarray, max_steps: int) -> np.ndarray:
    """Pad or truncate along time axis (axis=0) to exactly max_steps."""
    T, F = arr.shape
    if T >= max_steps:
        return arr[:max_steps]
    pad_width = max_steps - T
    return np.pad(arr, ((0, pad_width), (0, 0)), mode="constant")
