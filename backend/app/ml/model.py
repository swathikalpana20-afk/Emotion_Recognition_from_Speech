"""
EMOTISENSE AI — CNN + Bidirectional LSTM Model
Real deep learning architecture for speech emotion recognition.
Input: [batch, time_steps, feature_dim]
Output: [batch, num_classes] (log-softmax for training, softmax for inference)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── PyTorch (required) ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Model will not be available.")


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Run: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


class CNNBiLSTM(nn.Module):
    """
    CNN feature extractor → Bidirectional LSTM → Fully-connected classifier.

    Input shape:  [B, T, F]   (batch, time_steps, feature_dim)
    Output shape: [B, C]      (batch, num_classes) – raw logits
    """

    def __init__(
        self,
        input_dim: int = 120,
        num_classes: int = 8,
        cnn_channels: Tuple[int, int] = (64, 128),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        _require_torch()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # ── CNN block ────────────────────────────────────────────
        # Input: [B, 1, T, F] (treat feature map as image)
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, cnn_channels[0], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # halve frequency dim
            nn.Dropout2d(p=dropout / 2),
            # Conv block 2
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # halve again
            nn.Dropout2d(p=dropout / 2),
        )

        # Compute flattened CNN output size
        cnn_freq_out = input_dim // 4  # two MaxPool2d(1,2)
        self.cnn_out_dim = cnn_channels[1] * cnn_freq_out

        # ── Projection → LSTM input ───────────────────────────
        self.projection = nn.Linear(self.cnn_out_dim, lstm_hidden)

        # ── Bidirectional LSTM ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classifier head ───────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x: [B, T, F]
        returns: [B, num_classes] raw logits
        """
        batch_size, time_steps, feat_dim = x.shape

        # Add channel dim → [B, 1, T, F]
        x = x.unsqueeze(1)

        # CNN: [B, 1, T, F] → [B, C, T, F//4]
        x = self.cnn(x)

        # Reshape: [B, C, T, F//4] → [B, T, C * F//4]
        _, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, T2, C * F2)

        # Project: [B, T, C*F//4] → [B, T, lstm_hidden]
        x = self.projection(x)
        x = F.relu(x)

        # BiLSTM: [B, T, lstm_hidden] → [B, T, lstm_hidden*2]
        x, _ = self.lstm(x)

        # Attention-weighted mean (simple mean pooling for stability)
        x = x.mean(dim=1)  # [B, lstm_hidden*2]

        # Classify
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, num_classes]
        return logits


class CNNBaseline(nn.Module):
    """
    Simpler CNN-only baseline for comparison.
    Input: [B, T, F], Output: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 120,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        _require_torch()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=dropout / 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        B, T, F = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        return self.classifier(x)


# ── Model factory & checkpoint I/O ───────────────────────────────

def build_model(
    model_type: str = "cnn_bilstm",
    input_dim: int = 120,
    num_classes: int = 8,
    **kwargs,
) -> "nn.Module":
    _require_torch()
    if model_type == "cnn_bilstm":
        return CNNBiLSTM(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_type == "cnn_baseline":
        return CNNBaseline(input_dim=input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def save_checkpoint(
    model: "nn.Module",
    path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model state dict + metadata to .pt file."""
    _require_torch()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model_type: str = "cnn_bilstm",
    input_dim: int = 120,
    num_classes: int = 8,
    device: Optional[str] = None,
    **model_kwargs,
) -> Tuple["nn.Module", Dict]:
    """Load model from checkpoint file."""
    _require_torch()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model = build_model(model_type, input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded from {path} on {device}")
    return model, checkpoint.get("metadata", {})


def get_device() -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
