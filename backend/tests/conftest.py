"""conftest.py — shared fixtures for pytest."""
import io
import wave
import numpy as np
import pytest


def _make_sine_wav(duration=2.0, sample_rate=22050, freq=440.0):
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="session")
def sine_wav_bytes():
    return _make_sine_wav(duration=2.0)


@pytest.fixture(scope="session")
def short_wav_bytes():
    return _make_sine_wav(duration=0.2)


@pytest.fixture(scope="session")
def stereo_wav_bytes():
    n_samples = int(2.0 * 22050)
    t = np.linspace(0, 2.0, n_samples, endpoint=False)
    mono = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()
