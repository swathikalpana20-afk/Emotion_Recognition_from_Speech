"""
EMOTISENSE AI — Training Service
Manages asynchronous training jobs with progress tracking.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Job status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class TrainingJob:
    def __init__(self, job_id: str, dataset: str, config: Dict):
        self.job_id = job_id
        self.dataset = dataset
        self.config = config
        self.status = STATUS_PENDING
        self.progress_percent = 0.0
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 50)
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.val_accuracy: Optional[float] = None
        self.message = "Job queued."
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# In-memory job store (sufficient for single-process dev server)
_jobs: Dict[str, TrainingJob] = {}
_jobs_lock = threading.Lock()


def create_job(dataset: str, config: Dict) -> str:
    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(job_id, dataset, config)
    with _jobs_lock:
        _jobs[job_id] = job
    return job_id


def get_job(job_id: str) -> Optional[TrainingJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def start_training_job(job_id: str) -> None:
    """Run training in a background thread."""
    job = get_job(job_id)
    if job is None:
        return

    def _run():
        _train(job)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _train(job: TrainingJob) -> None:
    """Actual training logic (runs in background thread)."""
    from app.core.config import (
        ARTIFACTS_DIR, DATA_DIR, EMOTION_LABELS, RANDOM_SEED, REPORTS_DIR,
        VALIDATION_SPLIT, TEST_SPLIT,
    )
    from app.ml.datasets import load_dataset, UNIFIED_LABELS
    from app.ml.preprocessing import AudioPreprocessor, PreprocessingConfig
    from app.ml.model import build_model, save_checkpoint, get_device

    job.status = STATUS_RUNNING
    job.started_at = datetime.now(timezone.utc).isoformat()
    job.message = "Starting training..."

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import GroupShuffleSplit
        import numpy as np
    except ImportError as exc:
        job.status = STATUS_FAILED
        job.error = f"Missing dependency: {exc}"
        job.message = job.error
        return

    try:
        cfg = job.config
        epochs = cfg.get("epochs", 50)
        batch_size = cfg.get("batch_size", 32)
        lr = cfg.get("learning_rate", 1e-3)
        model_type = cfg.get("model_type", "cnn_bilstm")
        dataset_name = job.dataset
        seed = RANDOM_SEED

        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Load dataset paths
        dataset_dir = DATA_DIR / dataset_name
        job.message = f"Loading dataset {dataset_name}..."
        paths, labels_int, speakers = load_dataset(dataset_name, dataset_dir)
        labels_arr = np.array(labels_int)
        speakers_arr = np.array(speakers)

        if len(paths) < 10:
            raise RuntimeError(f"Too few samples ({len(paths)}). Check dataset directory.")

        # 2. Speaker-independent split
        unique_speakers = list(set(speakers))
        n_total = len(unique_speakers)
        n_test = max(1, int(n_total * TEST_SPLIT))
        n_val = max(1, int(n_total * VALIDATION_SPLIT))

        np.random.shuffle(unique_speakers)
        test_speakers = set(unique_speakers[:n_test])
        val_speakers = set(unique_speakers[n_test:n_test + n_val])

        train_idx = [i for i, s in enumerate(speakers) if s not in test_speakers and s not in val_speakers]
        val_idx = [i for i, s in enumerate(speakers) if s in val_speakers]
        test_idx = [i for i, s in enumerate(speakers) if s in test_speakers]

        if len(train_idx) < 5:
            # Fall back to random split if not enough speakers
            idx = np.arange(len(paths))
            np.random.shuffle(idx)
            n_test_s = max(1, int(len(idx) * TEST_SPLIT))
            n_val_s = max(1, int(len(idx) * VALIDATION_SPLIT))
            test_idx = idx[:n_test_s].tolist()
            val_idx = idx[n_test_s:n_test_s + n_val_s].tolist()
            train_idx = idx[n_test_s + n_val_s:].tolist()

        job.message = f"Dataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test"
        logger.info(job.message)

        # 3. Preprocessing fit on train split
        prep_cfg = PreprocessingConfig()
        preprocessor = AudioPreprocessor(prep_cfg)
        train_paths = [paths[i] for i in train_idx]
        job.message = "Fitting normalization statistics on training set..."
        preprocessor.fit(train_paths, max_samples=min(len(train_paths), 2000))

        # Save preprocessing artifacts
        prep_cfg.save(ARTIFACTS_DIR / "preprocessing_config.json")
        preprocessor.save_stats(ARTIFACTS_DIR / "norm_stats.npz")

        # 4. Extract features for all splits
        def extract_split(indices, name):
            feats, lbls = [], []
            for i in indices:
                try:
                    feat, _ = preprocessor.process_file(paths[i])
                    feats.append(feat)
                    lbls.append(labels_arr[i])
                except Exception as exc:
                    logger.warning(f"Skipping {paths[i].name}: {exc}")
            if not feats:
                raise RuntimeError(f"No valid features in {name} split.")
            return np.stack(feats), np.array(lbls)

        job.message = "Extracting features..."
        X_train, y_train = extract_split(train_idx, "train")
        X_val, y_val = extract_split(val_idx, "val")
        X_test, y_test = extract_split(test_idx, "test")

        # 5. DataLoaders
        device = get_device()
        def to_tensor_dataset(X, y):
            return TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )

        train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size)

        # 6. Model
        num_classes = len(UNIFIED_LABELS)
        model = build_model(model_type, input_dim=prep_cfg.feature_dim, num_classes=num_classes)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_epoch = 0
        patience_counter = 0
        PATIENCE = 7
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        # 7. Training loop
        job.total_epochs = epochs
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            # Validation
            model.eval()
            val_losses, val_correct, val_total = [], 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_losses.append(criterion(logits, yb).item())
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += len(yb)

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            job.current_epoch = epoch
            job.progress_percent = round(epoch / epochs * 100, 1)
            job.train_loss = train_loss
            job.val_loss = val_loss
            job.val_accuracy = val_acc
            job.message = (
                f"Epoch {epoch}/{epochs} — "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )
            logger.info(job.message)

            # Best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                metadata = {
                    "model_version": "1.0.0",
                    "preprocessing_version": "1.0.0",
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "num_classes": num_classes,
                    "input_dim": prep_cfg.feature_dim,
                    "best_val_accuracy": best_val_acc,
                    "best_epoch": best_epoch,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                }
                checkpoint_path = ARTIFACTS_DIR / "best_model.pt"
                save_checkpoint(model, checkpoint_path, metadata)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    job.message = f"Early stopping at epoch {epoch}."
                    logger.info(job.message)
                    break

        # 8. Evaluate on test set
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
        )

        checkpoint_path = ARTIFACTS_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            raise RuntimeError("Best model checkpoint not saved.")

        from app.ml.model import load_checkpoint
        best_model, _ = load_checkpoint(
            checkpoint_path, model_type=model_type,
            input_dim=prep_cfg.feature_dim, num_classes=num_classes,
        )
        best_model.eval()

        all_preds, all_true = [], []
        test_ds = to_tensor_dataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = best_model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        labels_list = UNIFIED_LABELS
        acc = accuracy_score(all_true, all_preds)
        macro_p = precision_score(all_true, all_preds, average="macro", zero_division=0)
        macro_r = recall_score(all_true, all_preds, average="macro", zero_division=0)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_true, all_preds, labels=list(range(num_classes))).tolist()
        report = classification_report(
            all_true,
            all_preds,
            labels=list(range(num_classes)),
            target_names=UNIFIED_LABELS[:num_classes],
            output_dict=True,
            zero_division=0,
        )
            
            
        

        eval_results = {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_class": {k: v for k, v in report.items() if k in labels_list},
            "confusion_matrix": cm,
            "dataset_name": dataset_name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "history": history,
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        job.status = STATUS_COMPLETED
        job.progress_percent = 100.0
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.message = (
            f"Training complete. Best val_acc={best_val_acc:.3f}, "
            f"Test accuracy={acc:.3f}, Macro F1={macro_f1:.3f}"
        )
        logger.info(job.message)

        # Reload the inference engine
        from app.services.inference import get_engine
        get_engine().load()

    except Exception as exc:
        job.status = STATUS_FAILED
        job.error = str(exc)
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.message = f"Training failed: {exc}"
        logger.exception("Training job failed", exc_info=exc)
