"""EMOTISENSE AI — Pydantic schemas for API request/response."""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: Dict[str, str]


class ModelStatusResponse(BaseModel):
    available: bool
    model_version: Optional[str] = None
    preprocessing_version: Optional[str] = None
    dataset_name: Optional[str] = None
    emotion_labels: List[str]
    num_classes: int
    sample_rate: int
    n_mfcc: int
    use_delta: bool
    use_delta2: bool
    feature_dim: int
    last_trained: Optional[str] = None
    message: str


class EmotionProbability(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: List[EmotionProbability]
    model_version: str
    preprocessing_version: str
    low_confidence_warning: bool
    silence_warning: bool
    mfcc_shape: List[int]
    audio_duration_seconds: float
    message: Optional[str] = None


class DatasetStatusResponse(BaseModel):
    dataset_name: str
    available: bool
    total_samples: int
    class_counts: Dict[str, int]
    missing_classes: List[str]
    message: str


class TrainRequest(BaseModel):
    dataset: str = Field(default="ravdess", description="Dataset name: ravdess, tess, emodb")
    epochs: Optional[int] = Field(default=None, ge=1, le=300)
    batch_size: Optional[int] = Field(default=None, ge=8, le=256)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    model_type: str = Field(default="cnn_bilstm", description="cnn_bilstm or cnn_baseline")


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress_percent: float
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    message: str
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    available: bool
    accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    per_class: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    dataset_name: Optional[str] = None
    evaluated_at: Optional[str] = None
    message: str


class InsightsResponse(BaseModel):
    available: bool
    mfcc_stats: Optional[Dict] = None
    class_distribution: Optional[Dict[str, int]] = None
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: int
