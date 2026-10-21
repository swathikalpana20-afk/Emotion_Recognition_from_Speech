# EMOTISENSE AI — Speech Emotion Recognition System

EMOTISENSE AI is an end-to-end Speech Emotion Recognition (SER) platform. It extracts acoustic features (MFCC + Delta + Delta-Delta) and classifies speech recordings using a deep learning architecture (hybrid CNN + Bidirectional LSTM). It includes a FastAPI backend, an interactive React (TypeScript + Vite) frontend, an automated test suite, dataset preparation tools, and standalone training scripts.

---

## 1. Project Structure

```
speach recoginition/
└── emotionai/
    ├── backend/                 # FastAPI backend application
    │   ├── app/
    │   │   ├── core/            # Configuration and hyperparameters
    │   │   ├── ml/              # Preprocessing (MFCC), PyTorch models, and dataset parsers
    │   │   ├── schemas/         # Pydantic request and response schemas
    │   │   ├── services/        # Audio inference and asynchronous background training
    │   │   ├── utils/           # Audio utilities and helpers
    │   │   └── main.py          # FastAPI application routes and middleware
    │   ├── tests/               # 32 automated unit and integration tests
    │   ├── pytest.ini           # Pytest configuration
    │   └── requirements.txt     # Python backend dependencies
    ├── frontend/                # React + TypeScript + Vite frontend
    │   ├── src/
    │   │   ├── components/      # UI components (Navbar, navigation)
    │   │   ├── pages/           # Dashboard, Analyze Studio, Model Insights, Training Lab
    │   │   ├── services/        # Axios API client
    │   │   ├── hooks/           # Custom audio recording and data fetching hooks
    │   │   ├── types/           # TypeScript API models
    │   │   ├── index.css        # Dark mode glassmorphism design system
    │   │   └── App.tsx          # Application router
    │   ├── package.json         # Frontend dependencies (React, Recharts, Lucide, Axios)
    │   └── vite.config.ts       # Vite bundler config with backend proxy
    ├── configs/                 # YAML configuration files (ravdess.yaml, tess.yaml)
    ├── scripts/                 # Standalone CLI tools (prepare_dataset.py, train.py)
    ├── docs/                    # Architecture and dataset documentation
    └── data/                    # Datasets folder (ravdess, tess, emodb)
```

---

## 2. Prerequisites

- **Python**: 3.10+ (tested on Python 3.11)
- **Node.js**: v18+ with `npm`
- **Operating System**: Windows (PowerShell commands provided below)

---

## 3. Installation & Setup

### A. Backend Setup
From the repository root, navigate to `emotionai/backend`:

```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai\backend"
pip install -r requirements.txt
```

### B. Frontend Setup
From the repository root, navigate to `emotionai/frontend`:

```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai\frontend"
npm install
```

---

## 4. Running Automated Tests

To run the complete automated test suite (verifying audio preprocessing, MFCC calculation, CNN + BiLSTM forward pass, API endpoints, and dataset handling):

```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai\backend"
pytest -v
```

*All 32 tests will run and pass.*

---

## 5. Starting the Application

To run the full stack, start both the backend and frontend servers in separate PowerShell terminals.

### Terminal 1: Start FastAPI Backend
```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **API Health Check**: `http://127.0.0.1:8000/api/health`

### Terminal 2: Start React Frontend
```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```
- **Web Application URL**: [http://127.0.0.1:5173/](http://127.0.0.1:5173/)

---

## 6. Dataset Preparation & Model Training

### Datasets Supported
1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):
   - Place files under `emotionai/data/ravdess/Actor_01/`, `Actor_02/`, etc.
2. **TESS** (Toronto Emotional Speech Set):
   - Place files under `emotionai/data/tess/OAF_angry/`, `YAF_happy/`, etc.

### Verify Dataset Layout
```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai"
python scripts/prepare_dataset.py --dataset ravdess
```

### Run Model Training
You can trigger training from the **Training Lab UI** at `http://127.0.0.1:5173/training` or via CLI:

```powershell
cd "c:\Users\Swathi Sri\OneDrive\Desktop\speach recoginition\emotionai"
python scripts/train.py --dataset ravdess --epochs 30 --batch-size 32 --model cnn_bilstm
```
Trained model checkpoints are automatically saved to `artifacts/best_model.pt` and evaluation metrics to `artifacts/evaluation_results.json`.

---

## 7. Features & Capabilities

- **Real-time Microphone Recording & File Upload**: In the **Analyze Studio**, record directly via your browser microphone or upload standard WAV files.
- **Genuine Emotion Predictions**: Outputs probability distribution across 8 classes (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`).
- **Model Insights**: Confusion matrix and per-class F1-score analytics.
- **Asynchronous Training Monitor**: Real-time epoch-by-epoch loss tracking in the frontend Training Lab.
