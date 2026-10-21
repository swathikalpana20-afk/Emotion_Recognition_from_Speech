"""
EMOTISENSE AI — FastAPI Application
All API routes: health, model status, predict, train, metrics, insights.
"""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ARTIFACTS_DIR,
    CORS_ORIGINS,
    DATA_DIR,
    EMOTION_LABELS,
    EVALUATION_RESULTS,
    FEATURE_DIM,
    MAX_TIME_STEPS,
    MAX_UPLOAD_SIZE_BYTES,
    MODEL_CHECKPOINT,
    MODEL_VERSION,
    N_MFCC,
    NUM_CLASSES,
    PREPROCESSING_VERSION,
    REPORTS_DIR,
    SAMPLE_RATE,
    USE_DELTA,
    USE_DELTA2,
)
from app.schemas.schemas import (
    DatasetStatusResponse,
    ErrorResponse,
    HealthResponse,
    InsightsResponse,
    MetricsResponse,
    ModelStatusResponse,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
)
from app.services import inference as inference_svc
from app.services import training as training_svc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="EMOTISENSE AI",
    description="Speech Emotion Recognition API — MFCC + CNN + BiLSTM",
    version=MODEL_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("EMOTISENSE AI backend starting up...")
    inference_svc.startup_load()


# ── Health ────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    deps = {}
    # Check librosa
    try:
        import librosa
        deps["librosa"] = librosa.__version__
    except ImportError:
        deps["librosa"] = "NOT INSTALLED"
    # Check torch
    try:
        import torch
        deps["torch"] = torch.__version__
    except ImportError:
        deps["torch"] = "NOT INSTALLED"
    # Check soundfile
    try:
        import soundfile as sf
        deps["soundfile"] = sf.__version__
    except ImportError:
        deps["soundfile"] = "NOT INSTALLED"

    return HealthResponse(
        status="ok",
        version=MODEL_VERSION,
        dependencies=deps,
    )


# ── Model status ──────────────────────────────────────────────────
@app.get("/api/model/status", response_model=ModelStatusResponse, tags=["Model"])
async def model_status():
    engine = inference_svc.get_engine()
    available = engine.is_loaded
    meta = engine.metadata

    if available:
        msg = "Model is loaded and ready for inference."
    else:
        msg = engine.load_error or "Model not trained. Train using POST /api/train."

    return ModelStatusResponse(
        available=available,
        model_version=meta.get("model_version", MODEL_VERSION) if available else None,
        preprocessing_version=meta.get("preprocessing_version", PREPROCESSING_VERSION) if available else None,
        dataset_name=meta.get("dataset") if available else None,
        emotion_labels=EMOTION_LABELS,
        num_classes=NUM_CLASSES,
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        use_delta=USE_DELTA,
        use_delta2=USE_DELTA2,
        feature_dim=FEATURE_DIM,
        last_trained=meta.get("trained_at") if available else None,
        message=msg,
    )


# ── Predict ───────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    # Validate extension
    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # Read with size limit
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed: {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB",
        )
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty file received.",
        )

    engine = inference_svc.get_engine()
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=engine.load_error or "Model not available. Train first.",
        )

    try:
        result = engine.predict(audio_bytes, filename=file.filename or "upload.wav")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return PredictionResponse(**result)


# ── Dataset status ────────────────────────────────────────────────
@app.get("/api/dataset/status", tags=["Dataset"])
async def dataset_status(dataset: str = "ravdess"):
    from app.ml.datasets import dataset_summary
    allowed = ["ravdess", "tess", "emodb"]
    if dataset.lower() not in allowed:
        raise HTTPException(400, detail=f"Unknown dataset '{dataset}'. Allowed: {allowed}")
    summary = dataset_summary(dataset.lower(), DATA_DIR / dataset.lower())
    return summary


# ── Train ─────────────────────────────────────────────────────────
@app.post("/api/train", response_model=TrainResponse, tags=["Training"])
async def start_train(req: TrainRequest):
    config = {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "model_type": req.model_type,
    }
    # Remove None values so training service uses defaults
    config = {k: v for k, v in config.items() if v is not None}
    config.setdefault("model_type", "cnn_bilstm")

    job_id = training_svc.create_job(req.dataset, config)
    training_svc.start_training_job(job_id)

    return TrainResponse(
        job_id=job_id,
        status="pending",
        message=f"Training job {job_id} started for dataset '{req.dataset}'.",
    )


# ── Train status ──────────────────────────────────────────────────
@app.get("/api/train/status/{job_id}", response_model=TrainStatusResponse, tags=["Training"])
async def train_status(job_id: str):
    job = training_svc.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return TrainStatusResponse(**job.to_dict())


# ── Metrics ───────────────────────────────────────────────────────
@app.get("/api/model/metrics", response_model=MetricsResponse, tags=["Model"])
async def model_metrics():
    if not EVALUATION_RESULTS.exists():
        return MetricsResponse(
            available=False,
            message="No evaluation results found. Train and evaluate the model first.",
        )
    try:
        with open(EVALUATION_RESULTS) as f:
            data = json.load(f)
        return MetricsResponse(
            available=True,
            accuracy=data.get("accuracy"),
            macro_precision=data.get("macro_precision"),
            macro_recall=data.get("macro_recall"),
            macro_f1=data.get("macro_f1"),
            per_class=data.get("per_class"),
            confusion_matrix=data.get("confusion_matrix"),
            dataset_name=data.get("dataset_name"),
            evaluated_at=data.get("evaluated_at"),
            message="Evaluation results loaded.",
        )
    except Exception as exc:
        return MetricsResponse(
            available=False,
            message=f"Could not load evaluation results: {exc}",
        )


# ── Insights ──────────────────────────────────────────────────────
@app.get("/api/insights", response_model=InsightsResponse, tags=["Insights"])
async def insights():
    result = {"available": False, "message": "No insights data available."}
    if EVALUATION_RESULTS.exists():
        try:
            with open(EVALUATION_RESULTS) as f:
                data = json.load(f)
            result = {
                "available": True,
                "class_distribution": data.get("class_distribution"),
                "message": "Insights loaded from training results.",
            }
        except Exception as exc:
            result["message"] = f"Error loading insights: {exc}"
    return result


# ── Error handlers ────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc), "code": 404},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": 500},
    )
