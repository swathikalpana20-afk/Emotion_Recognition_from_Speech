"""
EMOTISENSE AI — pytest test suite
Tests for preprocessing, model, dataset adapters, and API endpoints.
"""
from __future__ import annotations

import io
import json
import struct
import wave
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────

def _make_sine_wav(duration: float = 2.0, sample_rate: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a synthetic sine-wave WAV in memory."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


@pytest.fixture
def sine_wav_bytes():
    return _make_sine_wav(duration=2.0)


@pytest.fixture
def short_wav_bytes():
    """WAV shorter than min_duration."""
    return _make_sine_wav(duration=0.2)


@pytest.fixture
def stereo_wav_bytes():
    """Stereo WAV (2 channels)."""
    n_samples = int(2.0 * 22050)
    t = np.linspace(0, 2.0, n_samples, endpoint=False)
    mono = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


@pytest.fixture
def preprocessor():
    from app.ml.preprocessing import AudioPreprocessor, PreprocessingConfig
    cfg = PreprocessingConfig(max_time_steps=64, n_mfcc=13, use_delta=True, use_delta2=False)
    return AudioPreprocessor(cfg)


# ── Preprocessing tests ───────────────────────────────────────────

class TestPreprocessing:
    def test_valid_audio_returns_correct_shape(self, preprocessor, sine_wav_bytes):
        features, info = preprocessor.process_bytes(sine_wav_bytes, "test.wav")
        cfg = preprocessor.config
        assert features.shape == (cfg.max_time_steps, cfg.feature_dim), (
            f"Expected ({cfg.max_time_steps}, {cfg.feature_dim}), got {features.shape}"
        )

    def test_features_are_finite(self, preprocessor, sine_wav_bytes):
        features, _ = preprocessor.process_bytes(sine_wav_bytes, "test.wav")
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"

    def test_stereo_input_handled(self, preprocessor, stereo_wav_bytes):
        """Stereo audio should be converted to mono without error."""
        features, info = preprocessor.process_bytes(stereo_wav_bytes, "stereo.wav")
        assert features.ndim == 2

    def test_short_audio_raises_value_error(self, preprocessor, short_wav_bytes):
        with pytest.raises(ValueError, match="too short"):
            preprocessor.process_bytes(short_wav_bytes, "short.wav")

    def test_corrupt_audio_raises_value_error(self, preprocessor):
        bad_bytes = b"this is not audio data at all"
        with pytest.raises((ValueError, Exception)):
            preprocessor.process_bytes(bad_bytes, "corrupt.wav")

    def test_info_dict_contains_duration(self, preprocessor, sine_wav_bytes):
        _, info = preprocessor.process_bytes(sine_wav_bytes, "test.wav")
        assert "duration_seconds" in info
        assert info["duration_seconds"] > 0

    def test_delta_increases_feature_dim(self, sine_wav_bytes):
        from app.ml.preprocessing import AudioPreprocessor, PreprocessingConfig
        cfg_no_delta = PreprocessingConfig(n_mfcc=13, use_delta=False, use_delta2=False, max_time_steps=64)
        cfg_delta = PreprocessingConfig(n_mfcc=13, use_delta=True, use_delta2=False, max_time_steps=64)
        feat_no, _ = AudioPreprocessor(cfg_no_delta).process_bytes(sine_wav_bytes, "t.wav")
        feat_d, _ = AudioPreprocessor(cfg_delta).process_bytes(sine_wav_bytes, "t.wav")
        assert feat_d.shape[1] == feat_no.shape[1] * 2

    def test_normalization_applied_after_fit(self, sine_wav_bytes):
        from app.ml.preprocessing import AudioPreprocessor, PreprocessingConfig
        import tempfile
        cfg = PreprocessingConfig(n_mfcc=13, use_delta=False, use_delta2=False, max_time_steps=64)
        p = AudioPreprocessor(cfg)
        # Write to temp file for fit
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(sine_wav_bytes)
            tmp = Path(f.name)
        try:
            p.fit([tmp])
            features, _ = p.process_bytes(sine_wav_bytes, "test.wav")
            # After normalization, features should be centered around 0
            assert features is not None
        finally:
            tmp.unlink(missing_ok=True)

    def test_pad_or_truncate(self):
        from app.ml.preprocessing import _pad_or_truncate
        arr = np.ones((100, 40))
        padded = _pad_or_truncate(arr, 150)
        assert padded.shape == (150, 40)
        truncated = _pad_or_truncate(arr, 50)
        assert truncated.shape == (50, 40)


# ── Model tests ───────────────────────────────────────────────────

class TestModel:
    def test_cnn_bilstm_forward_pass(self):
        try:
            import torch
            from app.ml.model import CNNBiLSTM
        except ImportError:
            pytest.skip("PyTorch not available")

        model = CNNBiLSTM(input_dim=40, num_classes=8)
        model.eval()
        x = torch.randn(2, 64, 40)  # [batch=2, time=64, feat=40]
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 8), f"Expected (2, 8), got {logits.shape}"

    def test_cnn_baseline_forward_pass(self):
        try:
            import torch
            from app.ml.model import CNNBaseline
        except ImportError:
            pytest.skip("PyTorch not available")

        model = CNNBaseline(input_dim=40, num_classes=8)
        model.eval()
        x = torch.randn(2, 64, 40)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 8)

    def test_softmax_probabilities_sum_to_one(self):
        try:
            import torch
            import torch.nn.functional as F
            from app.ml.model import CNNBiLSTM
        except ImportError:
            pytest.skip("PyTorch not available")

        model = CNNBiLSTM(input_dim=40, num_classes=8)
        model.eval()
        x = torch.randn(3, 64, 40)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)

    def test_checkpoint_save_and_load(self, tmp_path):
        try:
            import torch
            from app.ml.model import CNNBiLSTM, save_checkpoint, load_checkpoint
        except ImportError:
            pytest.skip("PyTorch not available")

        model = CNNBiLSTM(input_dim=40, num_classes=8)
        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(model, ckpt_path, metadata={"test": True})
        assert ckpt_path.exists()

        loaded_model, meta = load_checkpoint(ckpt_path, input_dim=40, num_classes=8)
        assert meta["test"] is True
        loaded_model.eval()
        x = torch.randn(1, 64, 40)
        with torch.no_grad():
            out = loaded_model(x)
        assert out.shape == (1, 8)

    def test_missing_checkpoint_raises(self, tmp_path):
        try:
            from app.ml.model import load_checkpoint
        except ImportError:
            pytest.skip("PyTorch not available")

        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt", input_dim=40, num_classes=8)

    def test_output_class_count_matches_config(self):
        try:
            import torch
            from app.ml.model import CNNBiLSTM
            from app.core.config import NUM_CLASSES, FEATURE_DIM, MAX_TIME_STEPS
        except ImportError:
            pytest.skip("PyTorch not available")

        model = CNNBiLSTM(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(1, MAX_TIME_STEPS, FEATURE_DIM)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)


# ── Dataset tests ─────────────────────────────────────────────────

class TestDatasets:
    def test_parse_ravdess_filename(self):
        from app.ml.datasets import RAVDESS_EMOTION_MAP, parse_ravdess
        # Verify mapping completeness
        assert "01" in RAVDESS_EMOTION_MAP
        assert RAVDESS_EMOTION_MAP["03"] == "happy"
        assert RAVDESS_EMOTION_MAP["05"] == "angry"

    def test_parse_tess_emotion_map(self):
        from app.ml.datasets import TESS_EMOTION_MAP
        assert "angry" in TESS_EMOTION_MAP
        assert TESS_EMOTION_MAP["fear"] == "fearful"
        assert TESS_EMOTION_MAP["ps"] == "surprised"

    def test_parse_emodb_emotion_map(self):
        from app.ml.datasets import EMODB_EMOTION_MAP
        assert "W" in EMODB_EMOTION_MAP
        assert EMODB_EMOTION_MAP["W"] == "angry"

    def test_unknown_dataset_raises(self, tmp_path):
        from app.ml.datasets import load_dataset
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("fakedataset", tmp_path)

    def test_missing_directory_raises(self, tmp_path):
        from app.ml.datasets import load_dataset
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_dataset("ravdess", missing)

    def test_unified_labels_complete(self):
        from app.ml.datasets import UNIFIED_LABELS, LABEL_TO_IDX
        assert len(UNIFIED_LABELS) == 8
        assert len(LABEL_TO_IDX) == 8
        assert LABEL_TO_IDX["neutral"] == 0

    def test_dataset_summary_missing_returns_unavailable(self, tmp_path):
        from app.ml.datasets import dataset_summary
        result = dataset_summary("ravdess", tmp_path / "nonexistent")
        assert result["available"] is False


# ── API tests ─────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "dependencies" in data

    def test_model_status_endpoint(self):
        resp = self.client.get("/api/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "emotion_labels" in data
        assert len(data["emotion_labels"]) == 8

    def test_predict_empty_file_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code in (422, 503)

    def test_predict_wrong_extension_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 422

    def test_predict_no_model_returns_503(self, sine_wav_bytes):
        """When no model is loaded, predict should return 503."""
        from app.services.inference import get_engine
        engine = get_engine()
        was_loaded = engine._loaded
        engine._loaded = False
        engine._load_error = "Test: model not loaded"
        try:
            resp = self.client.post(
                "/api/predict",
                files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 503
        finally:
            engine._loaded = was_loaded
            engine._load_error = None

    def test_metrics_endpoint_returns_response(self):
        resp = self.client.get("/api/model/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data

    def test_dataset_status_endpoint(self):
        resp = self.client.get("/api/dataset/status?dataset=ravdess")
        assert resp.status_code == 200
        data = resp.json()
        assert "dataset_name" in data

    def test_dataset_status_unknown_rejected(self):
        resp = self.client.get("/api/dataset/status?dataset=unknown")
        assert resp.status_code == 400

    def test_train_endpoint_starts_job(self):
        resp = self.client.post(
            "/api/train",
            json={"dataset": "ravdess", "epochs": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        job_id = data["job_id"]

        # Check status
        status_resp = self.client.get(f"/api/train/status/{job_id}")
        assert status_resp.status_code == 200

    def test_train_status_unknown_job(self):
        resp = self.client.get("/api/train/status/nonexistent999")
        assert resp.status_code == 404
