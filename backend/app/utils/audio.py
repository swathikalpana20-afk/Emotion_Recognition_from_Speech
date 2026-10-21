"""
Audio helpers and utilities for EMOTISENSE AI.
"""
from __future__ import annotations

import io
import wave
import numpy as np


def generate_synthetic_audio(duration: float = 2.0, sample_rate: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a valid in-memory PCM 16-bit WAV audio byte array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())
    return buf.getvalue()
