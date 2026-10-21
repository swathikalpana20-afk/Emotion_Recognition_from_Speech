"""
EMOTISENSE AI — Core Configuration
All application-wide settings, paths, and model parameters.
"""
from pathlib import Path
import os

# ── Project roots ──────────────────────────────────────────────────
# File lives at: emotionai/backend/app/core/config.py
# 4 parents up = emotionai/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Ensure critical directories exist at import time
for _d in (ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model configuration ────────────────────────────────────────────
MODEL_CHECKPOINT = ARTIFACTS_DIR / "best_model.pt"
BASELINE_CHECKPOINT = ARTIFACTS_DIR / "baseline_cnn.pt"
EVALUATION_RESULTS = REPORTS_DIR / "evaluation_results.json"
PREPROCESSING_CONFIG = ARTIFACTS_DIR / "preprocessing_config.json"

# ── Audio preprocessing defaults ──────────────────────────────────
SAMPLE_RATE: int = 22050
N_MFCC: int = 40
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128
MAX_DURATION_SECONDS: float = 10.0
MIN_DURATION_SECONDS: float = 0.5
SILENCE_THRESHOLD_DB: float = -60.0
USE_DELTA: bool = True
USE_DELTA2: bool = True

# Fixed time steps for padding/truncation (approx 5 seconds)
MAX_TIME_STEPS: int = 216  # ceil(5 * 22050 / 512) = 216


def get_feature_dim() -> int:
    """Return MFCC feature dimension based on delta settings."""
    dim = N_MFCC
    if USE_DELTA:
        dim += N_MFCC
    if USE_DELTA2:
        dim += N_MFCC
    return dim


FEATURE_DIM: int = get_feature_dim()

# ── Emotion label mapping ─────────────────────────────────────────
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]
NUM_CLASSES: int = len(EMOTION_LABELS)

# ── Upload / inference limits ─────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a"}
TEMP_UPLOAD_DIR = Path(os.getenv("TEMP", "/tmp")) / "emotisense_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Training defaults ─────────────────────────────────────────────
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 7
RANDOM_SEED: int = 42
VALIDATION_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

# ── CNN + BiLSTM architecture defaults ────────────────────────────
CNN_CHANNELS = [64, 128]
LSTM_HIDDEN: int = 256
LSTM_LAYERS: int = 2
DROPOUT: float = 0.3

# ── API configuration ─────────────────────────────────────────────
API_VERSION: str = "v1"
MODEL_VERSION: str = "1.0.0"
PREPROCESSING_VERSION: str = "1.0.0"
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
