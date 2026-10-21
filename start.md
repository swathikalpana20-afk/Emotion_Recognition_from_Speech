# GOD MODE MASTER PROMPT V2.0
## Project: EMOTISENSE AI — Advanced Speech Emotion Recognition Platform

Act as a Principal AI/ML Engineer, Senior Python Developer, Deep Learning Researcher, Senior Full-Stack Engineer, UI/UX Designer, MLOps Engineer, QA Automation Engineer, and technical project mentor working together as one elite engineering team.

Your mission is to DESIGN, IMPLEMENT, TEST, DEBUG, DOCUMENT, and RUN a complete, polished, reproducible, competition-quality college project titled:

**“EMOTISENSE AI: An Intelligent Speech Emotion Recognition System Using MFCC, CNN and Bidirectional LSTM.”**

This must be a genuinely working end-to-end software project, not a static UI mockup, not a fake AI demo, and not a collection of disconnected code snippets.

Build the actual application inside the currently opened workspace. Inspect the workspace before making changes. Create the files, install or document appropriate dependencies, implement the application, run available tests, inspect failures, fix genuine issues, and verify the final workflow.

Do not stop after providing a plan. Continue implementation until the project reaches the strongest working state possible in the available environment.

---

# 1. PROJECT OBJECTIVE

Develop an intelligent speech emotion recognition platform that analyzes a person's recorded or uploaded speech audio and predicts the most likely emotion expressed in the audio.

The application must support:

- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised
- Neutral
- Calm

The exact class mapping must be configurable according to the selected dataset and the trained model.

Important scientific constraint: emotions inferred from voice are estimates of vocal expression, not reliable proof of a person's internal emotional state. Present predictions as model estimates and never as psychological diagnoses.

# 2. PRODUCT VISION

Build a product-quality, research-oriented application with the sophistication of a professional AI startup prototype.

The application should have:
- A premium, modern, responsive interface.
- A genuinely trained or properly loaded speech emotion model.
- Real audio preprocessing and feature extraction.
- Actual inference through the trained model.
- Interactive visualizations based on real predictions.
- Useful loading, empty, error, and success states.
- Clear setup instructions and reproducible execution.
- A demonstration mode that works without pretending an untrained model is accurate.

The project must be impressive during a live college demonstration and easy for an evaluator to understand.

# 3. REQUIRED TECHNOLOGY STACK

Prefer a simple, stable architecture with minimal unnecessary dependencies.

Machine learning:
- Python 3.11 where compatible with the environment.
- PyTorch for deep learning.
- Librosa for audio loading and MFCC extraction.
- NumPy and scikit-learn for data processing, splitting, and evaluation.
- SoundFile for audio file handling.
- Matplotlib or Plotly for visualizations.

Backend:
- FastAPI.
- Pydantic validation.
- Uvicorn.
- Structured logging and consistent API error responses.

Frontend:
- React with TypeScript and Vite, if the workspace supports a separate frontend.
- A clean component-based architecture.
- A modern CSS system or Tailwind CSS if setup is reliable.
- Recharts for interactive charts, if appropriate.
- Browser MediaRecorder API for microphone recording.

Testing:
- pytest for backend and ML tests.
- Vitest and React Testing Library if a React frontend is used.
- Playwright only if practical in the available environment.

Architecture rules:
- Prefer a maintainable monorepo.
- Avoid unnecessary microservices.
- Pin compatible dependency versions.
- Never introduce dependencies without a clear purpose.
- If the environment cannot support the preferred stack, select the closest reliable alternative and document the decision.

# 4. REQUIRED APPLICATION MODULES

## A. Premium Dashboard

Build a professional landing dashboard with:
- Product name and logo treatment.
- Project description.
- A clear audio analysis call to action.
- Dataset and model status.
- Model readiness indicator.
- Recent analyses when history is enabled.
- Emotion distribution analytics when real data exists.
- Clear navigation to Analyze, Insights, Model Information, and About.

Never display fabricated accuracy, prediction history, user counts, or training metrics.

When no model is trained, clearly show “Model not trained” and guide the user to train it or use an explicitly labeled demo mode.

## B. Audio Analysis Studio

Implement two real input methods:

1. Upload audio:
   - WAV support as the reliable default.
   - Support additional formats only when decoding dependencies are installed.
   - Validate file size, duration, sample rate, and readable audio data.
   - Reject empty, corrupt, excessively long, or unsupported files gracefully.

2. Microphone recording:
   - Request microphone permission through the browser.
   - Record audio with MediaRecorder.
   - Show recording duration and recording state.
   - Provide stop, playback, and retry controls.
   - Send the recorded audio to the backend for genuine inference.
   - Handle denied permissions, unsupported browsers, and recording failures.

Show an audio waveform or meaningful audio preview if feasible.

Do not claim that live inference is occurring while the user is recording unless streaming inference is actually implemented. A normal recording workflow may analyze the audio after the user presses Stop and Analyze.

## C. Audio Preprocessing Pipeline

Implement a shared, reusable preprocessing pipeline for training and inference.

Required steps:
1. Decode the audio safely.
2. Convert to mono where appropriate.
3. Resample to the model's configured sample rate.
4. Apply consistent amplitude handling.
5. Handle silence and near-silent recordings.
6. Apply a documented duration policy.
7. Extract MFCC features using Librosa.
8. Generate the same feature dimensions expected by the model.
9. Normalize features using training-only statistics or another documented training procedure.
10. Validate tensor shape, numeric values, and finite outputs.

Use configurable parameters such as:
- Sample rate: 22,050 Hz, unless dataset/model compatibility calls for another value.
- MFCC count: 40.
- Window and hop lengths chosen consistently.
- Optional delta and delta-delta coefficients.
- Padding or truncation for variable-length sequences.

Do not independently normalize training and inference samples in a way that introduces a train/inference mismatch.

Make preprocessing deterministic wherever possible.

## D. Deep Learning Model

Implement a real CNN + Bidirectional LSTM architecture.

Recommended architecture:
- Input: time-frequency MFCC feature sequence.
- CNN feature extractor for local acoustic patterns.
- Appropriate tensor reshaping and feature projection.
- Bidirectional LSTM for temporal context.
- Dropout for regularization.
- Fully connected classification layer.
- Softmax probabilities for inference.

The exact architecture may be refined based on dataset size and measured performance.

Requirements:
- Correct tensor dimensions.
- Correct number of output classes.
- Proper device selection for CPU or supported GPU.
- Reproducible random seeds where practical.
- Cross-entropy training objective.
- Adam or AdamW optimizer.
- Configurable batch size, learning rate, and epoch count.
- Validation monitoring.
- Early stopping or best-checkpoint selection.
- Model checkpoint saving and safe loading.
- Clear model version and preprocessing configuration.
- Unit tests for forward pass, output dimensions, and probability normalization.

Also support a simpler CNN baseline where practical, so the hybrid model can be compared against a baseline using the same evaluation protocol.

Never report benchmark superiority without actually measuring it.

## E. Dataset Support

Support the following public speech emotion datasets through configurable dataset adapters:

1. RAVDESS.
2. TESS.
3. EMO-DB.

Provide dataset-specific filename parsing and label mapping based on official dataset conventions.

Requirements:
- Do not bundle copyrighted or large datasets without authorization.
- Do not assume the datasets are already installed.
- Provide exact expected folder layouts.
- Provide dataset preparation and validation scripts.
- Detect missing folders, missing classes, malformed files, and corrupted audio.
- Generate a dataset summary containing class counts, total usable samples, exclusions, and reasons for exclusion.
- Support training on one dataset at a time.
- Allow combined datasets only through an explicit configuration and consistent label mapping.

Implement the RAVDESS emotion code mapping using the official naming convention, including calm, happy, sad, angry, fearful, disgust, and surprised. Include neutral when present in the chosen dataset configuration.

Implement appropriate TESS and EMO-DB mappings based on their official label conventions.

Never silently map unknown labels to a valid emotion.

## F. Leakage-Resistant Training and Evaluation

Use a documented train/validation/test split.

Where speaker metadata exists, use speaker-independent splits or grouped cross-validation so the same speaker does not appear across training and test sets.

Where multiple recordings are derived from the same original utterance, avoid allowing related samples to leak across splits.

Do not fit feature normalization, feature selection, or any other learned preprocessing on the test set.

Report:
- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Per-class precision, recall, and F1.
- Confusion matrix.
- Training and validation loss curves.
- Training and validation metric curves when available.

If the dataset is too small or a class is missing, clearly document the limitation.

Do not invent training results. Clearly distinguish untrained, partially trained, and fully evaluated states.

## G. Emotion Prediction Results

After successful inference, display:
- Predicted emotion.
- Confidence score.
- Probability distribution for every supported class.
- Interactive probability chart.
- Audio playback.
- MFCC visualization.
- Prediction timestamp if stored.
- Model version and preprocessing version.

Use calibrated wording such as “Predicted emotion: Happy” and “Model confidence: 72%.”

Explain that confidence is not the probability that the person truly feels that emotion, and that uncalibrated softmax scores may be overconfident.

Display a clear warning when confidence is low or when the recording contains too little speech.

Do not force a confident label for silence, corrupted audio, or inputs outside the model's supported conditions.

An optional low-confidence or “uncertain” outcome may be added through a documented validation-based threshold. Do not train or present “uncertain” as a real class unless the dataset and model explicitly support it.

## H. Advanced Insights and Explainability

Implement meaningful visual analysis:
- MFCC heatmap.
- Audio waveform.
- Class probability chart.
- Confusion matrix from real evaluation results.
- Class-wise F1 chart.
- Dataset class distribution.
- Training and validation loss curves.

Where useful, add feature-space visualization using PCA or t-SNE, clearly labeled as exploratory.

All plots must use real computed data.

Explain that MFCCs summarize aspects of the speech spectrum and do not directly reveal a person's emotions.

Avoid unsupported claims that individual MFCC coefficients independently explain a particular emotion.

## I. Model Information and Research Panel

Create a page describing:
- Problem statement.
- Objectives.
- Dataset sources and license considerations.
- MFCC feature extraction.
- CNN feature learning.
- Bidirectional LSTM temporal modeling.
- Training and evaluation process.
- API architecture.
- Known limitations.
- Ethical considerations.
- Future enhancements.

Include a model status card showing:
- Model availability.
- Dataset name.
- Supported classes.
- Sample rate.
- MFCC count.
- Model version.
- Last successful training timestamp, when available.
- Evaluation metrics, only when backed by saved evaluation results.

# 5. PROFESSIONAL UI/UX DESIGN SYSTEM

Design a distinctive premium interface inspired by professional audio software, AI research dashboards, and polished SaaS products.

Visual direction:
- Deep navy or charcoal background.
- Restrained electric cyan, violet, or teal accents.
- High contrast and excellent text readability.
- Subtle gradients.
- Consistent spacing.
- Clear typography hierarchy.
- Rounded cards used purposefully.
- Subtle motion and hover states.
- Elegant charts and audio visualizations.
- Responsive desktop, tablet, and mobile layouts.

Create a custom EMOTISENSE AI brand identity.

Required screens:
1. Dashboard.
2. Audio analysis studio.
3. Prediction results.
4. Model insights.
5. Dataset and training status.
6. About and technical documentation.

UX requirements:
- Accessible labels.
- Keyboard-friendly controls.
- Visible focus states.
- Responsive layout.
- Loading indicators.
- Empty states.
- Success and error notifications.
- Confirmation for destructive operations.
- Useful inline validation.
- No fake controls.
- No dead links.
- No placeholder buttons presented as completed features.

The UI must remain usable even when the backend is unavailable.

# 6. BACKEND API CONTRACT

Implement a clean REST API with documented request and response schemas.

Suggested endpoints:

GET /api/health
Returns backend health and dependency status.

GET /api/model/status
Returns whether a trained model is available and its metadata.

POST /api/predict
Accepts validated audio and returns genuine model predictions.

GET /api/model/metrics
Returns saved evaluation metrics when available.

GET /api/dataset/status
Returns dataset availability and preparation status.

POST /api/train
Starts a training job only when the required datasets and training configuration are valid. Return a job identifier and actual job status.

GET /api/train/status/{job_id}
Returns training progress, logs summary, and checkpoint status.

GET /api/insights
Returns actual persisted analytics when supported.

Use appropriate HTTP status codes and clear JSON error messages.

Configure CORS narrowly for the intended frontend origin.

Add request validation, sensible upload limits, timeouts where appropriate, and safe error handling.

Never expose stack traces, secrets, local absolute paths, or internal configuration in public API responses.

Avoid accepting arbitrary filesystem paths from API clients.

Document the API through FastAPI's OpenAPI documentation.

If model training is not configured, return a clear explanatory status rather than pretending the training succeeded.

# 7. SECURITY, PRIVACY, AND RELIABILITY

Treat uploaded audio as potentially sensitive data.

Requirements:
- Process audio locally by default.
- Do not send audio to external AI services.
- Do not retain raw recordings unless the user explicitly enables an explained storage option.
- Delete temporary audio files after processing where feasible.
- Validate uploads before decoding.
- Use generated temporary filenames rather than user-supplied paths.
- Prevent path traversal.
- Set upload size and duration limits.
- Avoid shell command construction from user input.
- Never commit credentials or secrets.
- Add .env.example with safe placeholder values.
- Exclude datasets, recordings, model checkpoints, caches, and environment folders from version control where appropriate.
- Document all storage behavior.

Add robust error handling for:
- Missing model weights.
- Missing datasets.
- Corrupt audio.
- Unsupported sample formats.
- NaN or infinite features.
- Incorrect tensor shapes.
- Model loading failures.
- Backend startup failures.
- Frontend API failures.
- Microphone permission denial.
- Empty recordings.
- Insufficient disk space where detectable.

# 8. TESTING REQUIREMENTS

Create meaningful automated tests for:

Preprocessing:
- Valid audio.
- Mono and stereo inputs.
- Resampling.
- Silence detection.
- Invalid and corrupted audio.
- Feature shape and finite values.

Model:
- Forward pass.
- Output class count.
- Probability sums.
- Checkpoint save/load.
- Missing checkpoint handling.

Dataset:
- Filename parsing.
- Label mapping.
- Missing dataset detection.
- Unknown-label rejection.
- Split integrity.

API:
- Health endpoint.
- Model status.
- Invalid uploads.
- Successful prediction using a real test checkpoint.
- Missing-model behavior.
- Training status.

Frontend:
- Dashboard rendering.
- File upload validation.
- Recording controls.
- Loading and error states.
- Prediction result rendering.

Do not rely exclusively on mocked predictions to claim end-to-end success.

Create a deterministic synthetic-audio fixture for preprocessing and API error-path tests. Do not use synthetic audio as proof of emotion classification accuracy.

Run all feasible tests and fix failures caused by the implementation.

If a test cannot run because of unavailable system dependencies or missing datasets, report the exact limitation.

# 9. TRAINING PIPELINE AND COMMANDS

Provide reproducible scripts and documented commands for:

1. Creating the Python environment.
2. Installing dependencies.
3. Preparing datasets.
4. Validating dataset labels.
5. Training the CNN baseline.
6. Training the CNN + BiLSTM model.
7. Evaluating a saved checkpoint.
8. Starting the backend.
9. Starting the frontend.
10. Running automated tests.

Example command design:

python -m src.data.prepare --config configs/ravdess.yaml

python -m src.training.train --config configs/ravdess.yaml

python -m src.evaluation.evaluate --checkpoint artifacts/best_model.pt

python -m pytest

npm run dev

Adapt commands to the actual implemented package structure. Never document commands that do not exist.

Provide:
- requirements.txt or an appropriate Python project configuration.
- Frontend package.json.
- .env.example.
- .gitignore.
- Dataset configuration files.
- README.md.
- Model card.
- Training and evaluation documentation.

# 10. PROJECT STRUCTURE

Create a logical and maintainable structure similar to:

emotisense-ai/
  backend/
    app/
      main.py
      api/
      core/
      schemas/
      services/
      ml/
      utils/
    tests/
    requirements.txt
  frontend/
    src/
      components/
      pages/
      hooks/
      services/
      types/
      styles/
    package.json
  configs/
  scripts/
  artifacts/
  reports/
  docs/
  .env.example
  .gitignore
  README.md

Adapt the structure when necessary, but maintain clear separation between frontend, backend, dataset processing, training, inference, and tests.

Do not create empty directories merely to imitate this layout.

# 11. ENGINEERING WORKFLOW

Follow this sequence:

PHASE 1 — INSPECT
Inspect the workspace, existing source files, runtime versions, and available dependencies. Preserve useful existing work.

PHASE 2 — ARCHITECT
Define the architecture, API contract, model input shape, emotion-label mapping, and data flow.

PHASE 3 — IMPLEMENT
Build the backend, frontend, preprocessing pipeline, model, dataset adapters, training scripts, and documentation.

PHASE 4 — EXECUTE
Install only required dependencies when permitted. Start services using the available terminal tools.

PHASE 5 — TEST
Run automated tests, validate imports, inspect logs, and test actual API behavior.

PHASE 6 — DEBUG
Investigate failures using their real error messages. Fix root causes rather than suppressing exceptions or hiding errors.

PHASE 7 — VERIFY
Verify the following user journey:
1. Open the application.
2. View model status.
3. Upload an audio file or record speech.
4. Submit the audio.
5. Run the actual trained model if available.
6. View prediction probabilities.
7. View the MFCC visualization.
8. Recover gracefully from an invalid file or unavailable model.

PHASE 8 — DOCUMENT
Update the README to reflect the exact implementation, dependencies, dataset setup, commands, tests, and known limitations.

PHASE 9 — FINAL REPORT
Provide a concise engineering handoff including:
- Features actually implemented.
- Important files created.
- Architecture summary.
- Commands to run the application.
- Dataset preparation instructions.
- Training and evaluation instructions.
- Tests executed and their real results.
- Any remaining errors or environment limitations.
- Clear next steps for obtaining genuine evaluation results.

# 12. STRICT QUALITY GATES

These rules are mandatory:

1. Do not stop after generating a plan.
2. Do not generate only a static frontend.
3. Do not hardcode a prediction and present it as machine learning.
4. Do not fabricate accuracy, training history, datasets, or evaluation results.
5. Do not claim that a model is trained when only random initialization exists.
6. Do not silently substitute a mock backend for the real backend.
7. Do not hide exceptions with broad catch blocks that make failures look successful.
8. Do not suppress errors merely to make the application appear functional.
9. Do not silently skip failing tests.
10. Do not assume a package is installed without verifying it.
11. Do not download large datasets automatically without permission.
12. Do not overwrite existing user work unnecessarily.
13. Do not introduce unnecessary complexity.
14. Do not leave primary navigation, buttons, or forms nonfunctional.
15. Do not claim full verification if the available environment cannot run it.
16. Clearly separate implemented features, verified features, and planned features.
17. Prefer small, testable modules over one enormous source file.
18. Make the smallest justified changes when fixing bugs.
19. Re-run relevant tests after making fixes.
20. Continue working through the implementation and verification steps until completion or a genuine environment blocker is reached.

# 13. COMPETITION-READY ENHANCEMENTS

After the core system works, implement the following in priority order:

Priority 1:
- Real model inference.
- Reliable audio preprocessing.
- Strong input validation.
- Responsive, polished interface.
- Clear evaluation metrics.

Priority 2:
- Interactive MFCC heatmap.
- Training progress and model comparison.
- Speaker-independent evaluation.
- Exportable analysis report containing prediction results and model metadata.
- A comparison between the CNN baseline and CNN + BiLSTM.

Priority 3:
- Optional multilingual dataset experiments.
- Confidence calibration.
- Carefully validated noise robustness experiments.
- Optional ONNX inference if it offers a real deployment benefit.

Any enhancement must be implemented and tested before being marked complete.

Do not add sentiment-from-text or speech transcription as if they were part of the core emotion classifier. If added later, treat them as separate optional modules.

# 14. FINAL ACCEPTANCE CRITERIA

The project is complete only to the extent that the following criteria have been verified:

- The frontend starts successfully.
- The backend starts successfully.
- The application can communicate with the backend.
- Audio uploads are validated.
- MFCC features are computed correctly.
- The CNN + BiLSTM model can execute a forward pass.
- A trained checkpoint can be loaded for actual inference.
- Predictions are returned using the correct class mapping.
- The results screen displays genuine model outputs.
- Evaluation scripts compute metrics from real held-out data.
- Important automated tests pass.
- Missing datasets and model files produce useful error messages.
- Setup and usage instructions match the actual project.
- Limitations are documented honestly.

A fully trained and scientifically evaluated model requires access to the relevant dataset and successful training. If either is unavailable, finish all other feasible work and explain exactly what remains.

Start now by inspecting the current workspace, then implement the complete project in stages. Make sound engineering decisions independently, preserve existing work, test continuously, fix errors, and provide a truthful final implementation report.

Your objective is not to produce the most code. Your objective is to deliver the most reliable, impressive, scientifically honest, and demonstrably functional version of EMOTISENSE AI that the available environment can support.