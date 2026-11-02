"""Small audio helpers shared by the generators and the systems under test."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
FRAME_S = 0.010  # analysis hop for ground-truth labelling


def read(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def write(path: str, x: np.ndarray, sr: int) -> None:
    sf.write(path, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def frame_db(x: np.ndarray, sr: int, hop_s: float = FRAME_S) -> np.ndarray:
    """RMS level per non-overlapping frame, in dB (re. full scale)."""
    hop = int(round(hop_s * sr))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1)) + 1e-10
    return 20 * np.log10(rms)


def trim(x: np.ndarray, sr: int, rel_db: float = -45.0) -> np.ndarray:
    """Trim leading/trailing low-level audio (relative to the loudest frame)."""
    db = frame_db(x, sr)
    if len(db) == 0:
        return x
    on = np.where(db > db.max() + rel_db)[0]
    if len(on) == 0:
        return x[:0]
    hop = int(round(FRAME_S * sr))
    return x[on[0] * hop : (on[-1] + 1) * hop]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximately 1/f noise (Voss-McCartney style via spectral shaping)."""
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    f = f / np.sqrt(k)
    y = np.fft.irfft(f, n)
    return (y / (np.std(y) + 1e-12)).astype(np.float32)


def telephony(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Narrowband telephony simulation: 300-3400 Hz band-pass, 8 kHz, G.711 mu-law.

    Returns the 8 kHz signal.
    """
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x).astype(np.float32)
    y = resample(y, sr, 8000)
    mu = 255.0
    y = np.clip(y, -1, 1)
    enc = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * 255) / 255 * 2 - 1  # 8-bit quantisation
    dec = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return dec.astype(np.float32)
