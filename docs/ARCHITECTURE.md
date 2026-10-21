# EMOTISENSE AI Documentation

EMOTISENSE AI is an end-to-end Speech Emotion Recognition (SER) system that combines acoustic feature extraction (MFCC + Delta + Delta-Delta) with a hybrid Deep Learning architecture (CNN + Bidirectional LSTM).

## Architecture Overview

```
Speech Audio (.wav)
       │
       ▼
[Audio Preprocessing]
  ├── Resample (22,050 Hz)
  ├── Trim silence / Normalize
  └── Extract MFCC (40 coefficients) + Δ + ΔΔ (Total 120 features)
       │
       ▼
[Deep Learning Model]
  ├── Conv2D Layers (Local acoustic feature maps)
  ├── MaxPooling2D + BatchNormalization + SpatialDropout
  ├── Bidirectional LSTM (Temporal sequence context)
  ├── Dense Layers + Dropout
  └── Softmax (8 Emotion Classes)
       │
       ▼
[Inference & Confidence Scoring]
  ├── Primary Emotion Classification
  └── Probabilistic Distribution across all classes
```

## Supported Emotion Classes
1. **Neutral**
2. **Calm**
3. **Happy**
4. **Sad**
5. **Angry**
6. **Fearful**
7. **Disgust**
8. **Surprised**

## API Endpoints
- `GET /api/health` — Backend health check & dependency status.
- `GET /api/model/status` — Model availability, version, and architecture info.
- `POST /api/predict` — Upload WAV audio file for emotion recognition.
- `GET /api/dataset/status?dataset={name}` — Dataset readiness check.
- `POST /api/train` — Start asynchronous model training on chosen dataset.
- `GET /api/train/status/{job_id}` — Poll training job progress & epoch logs.
- `GET /api/model/metrics` — Accuracy, Macro F1, Precision, Recall, and Confusion Matrix.
