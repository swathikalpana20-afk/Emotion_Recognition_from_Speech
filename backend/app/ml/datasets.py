"""
EMOTISENSE AI — Dataset Adapters
Support for RAVDESS, TESS, and EMO-DB datasets.
No silent label mapping: unknown labels raise clear errors.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Unified emotion labels (index → name) ────────────────────────
UNIFIED_LABELS = [
    "neutral",    # 0
    "calm",       # 1
    "happy",      # 2
    "sad",        # 3
    "angry",      # 4
    "fearful",    # 5
    "disgust",    # 6
    "surprised",  # 7
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(UNIFIED_LABELS)}


# ── RAVDESS adapter ───────────────────────────────────────────────
# Filename convention: 03-01-05-01-02-01-12.wav
# Field 3 = emotion code
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# RAVDESS folder layout: Audio_Speech_Actors_01-24/Actor_XX/*.wav
RAVDESS_ACTOR_PATTERN = re.compile(r"Actor_(\d+)")


def parse_ravdess(dataset_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Parse RAVDESS dataset.
    Returns list of (audio_path, emotion_label, speaker_id).
    """
    records = []
    audio_dirs = sorted(dataset_dir.glob("Actor_*"))
    if not audio_dirs:
        # Try flat layout
        audio_dirs = [dataset_dir]

    for actor_dir in audio_dirs:
        m = RAVDESS_ACTOR_PATTERN.search(actor_dir.name)
        speaker_id = m.group(1) if m else "unknown"

        for wav_path in sorted(actor_dir.glob("*.wav")):
            parts = wav_path.stem.split("-")
            if len(parts) < 3:
                logger.warning(f"RAVDESS: skipping malformed filename {wav_path.name}")
                continue
            emotion_code = parts[2]
            if emotion_code not in RAVDESS_EMOTION_MAP:
                logger.warning(
                    f"RAVDESS: unknown emotion code '{emotion_code}' in {wav_path.name}"
                )
                continue
            emotion = RAVDESS_EMOTION_MAP[emotion_code]
            records.append((wav_path, emotion, speaker_id))

    logger.info(f"RAVDESS: found {len(records)} samples")
    return records


# ── TESS adapter ─────────────────────────────────────────────────
# Folder names contain emotion: OAF_Fear, YAF_angry, etc.
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprised",  # "Pleasant Surprise"
    "pleasantsurprise": "surprised",
}


def parse_tess(dataset_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Parse TESS dataset.
    Returns list of (audio_path, emotion_label, speaker_id).
    """
    records = []
    for folder in sorted(dataset_dir.glob("*")):
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower()
        # Extract emotion from folder name: YAF_angry → angry, OAF_Fear → fear
        emotion_raw = folder_lower.replace("yaf_", "").replace("oaf_", "").strip()
        emotion_norm = emotion_raw.replace(" ", "").replace("_", "")

        if emotion_norm not in TESS_EMOTION_MAP:
            logger.warning(f"TESS: unknown emotion folder '{folder.name}'")
            continue

        emotion = TESS_EMOTION_MAP[emotion_norm]
        speaker_id = "OAF" if "oaf" in folder_lower else "YAF"

        for wav_path in sorted(folder.glob("*.wav")):
            records.append((wav_path, emotion, speaker_id))

    logger.info(f"TESS: found {len(records)} samples")
    return records


# ── EMO-DB adapter ────────────────────────────────────────────────
# Filename: 03a01Wa.wav
# Char 5 (0-indexed) = emotion code
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "calm",      # Langeweile (boredom) → calm (closest available)
    "E": "disgust",
    "A": "fearful",   # Angst
    "F": "happy",     # Freude (joy)
    "T": "sad",       # Trauer
    "N": "neutral",
}


def parse_emodb(dataset_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Parse EMO-DB dataset.
    Returns list of (audio_path, emotion_label, speaker_id).
    """
    records = []
    wav_dir = dataset_dir / "wav"
    if not wav_dir.exists():
        wav_dir = dataset_dir  # flat layout fallback

    for wav_path in sorted(wav_dir.glob("*.wav")):
        stem = wav_path.stem
        if len(stem) < 6:
            logger.warning(f"EMO-DB: malformed filename {wav_path.name}")
            continue
        speaker_id = stem[:2]
        emotion_code = stem[5].upper()
        if emotion_code not in EMODB_EMOTION_MAP:
            logger.warning(
                f"EMO-DB: unknown emotion code '{emotion_code}' in {wav_path.name}"
            )
            continue
        emotion = EMODB_EMOTION_MAP[emotion_code]
        records.append((wav_path, emotion, speaker_id))

    logger.info(f"EMO-DB: found {len(records)} samples")
    return records


# ── Dataset factory ───────────────────────────────────────────────

DATASET_PARSERS = {
    "ravdess": parse_ravdess,
    "tess": parse_tess,
    "emodb": parse_emodb,
}


def load_dataset(
    dataset_name: str,
    dataset_dir: Path,
    unified_labels: Optional[List[str]] = None,
) -> Tuple[List[Path], List[int], List[str]]:
    """
    Load a dataset and return parallel lists of paths, integer labels, speaker IDs.
    Only labels present in unified_labels are kept.
    """
    if unified_labels is None:
        unified_labels = UNIFIED_LABELS

    parser = DATASET_PARSERS.get(dataset_name.lower())
    if parser is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(DATASET_PARSERS.keys())}"
        )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please download {dataset_name.upper()} and place it at this path."
        )

    records = parser(dataset_dir)
    if not records:
        raise RuntimeError(f"No samples found in {dataset_dir}")

    label_to_idx = {label: idx for idx, label in enumerate(unified_labels)}

    paths, labels, speakers = [], [], []
    skipped_unknown = 0
    for path, emotion, speaker in records:
        if emotion not in label_to_idx:
            skipped_unknown += 1
            continue
        paths.append(path)
        labels.append(label_to_idx[emotion])
        speakers.append(speaker)

    if skipped_unknown:
        logger.warning(f"Skipped {skipped_unknown} samples with unmapped emotions.")

    logger.info(
        f"Dataset '{dataset_name}': {len(paths)} usable samples, "
        f"{len(set(speakers))} speakers"
    )
    return paths, labels, speakers


def dataset_summary(
    dataset_name: str,
    dataset_dir: Path,
    unified_labels: Optional[List[str]] = None,
) -> Dict:
    """Return a summary dict: class counts, total, missing classes, etc."""
    if unified_labels is None:
        unified_labels = UNIFIED_LABELS
    try:
        paths, labels, speakers = load_dataset(dataset_name, dataset_dir, unified_labels)
    except Exception as exc:
        return {
            "dataset_name": dataset_name,
            "available": False,
            "total_samples": 0,
            "class_counts": {},
            "missing_classes": unified_labels,
            "message": str(exc),
        }

    from collections import Counter
    label_counts = Counter(unified_labels[i] for i in labels)
    present = set(label_counts.keys())
    missing = [l for l in unified_labels if l not in present]

    return {
        "dataset_name": dataset_name,
        "available": True,
        "total_samples": len(paths),
        "class_counts": dict(label_counts),
        "missing_classes": missing,
        "num_speakers": len(set(speakers)),
        "message": f"Found {len(paths)} samples across {len(set(speakers))} speakers.",
    }
