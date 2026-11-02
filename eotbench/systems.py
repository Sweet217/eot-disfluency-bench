"""Endpointers under test.

All systems share the same causal decision structure: they watch a stream of
VAD frames and fire once. Per-item VAD output and per-pause model scores are
computed once (see scripts/extract.py) and then replayed here, so threshold
sweeps are cheap.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from .vad import SILERO_FRAME_S

# ------------------------------------------------------------ silence threshold

def silence_fire(runs: list[tuple[int, int]], frame_s: float, threshold_ms: float) -> float | None:
    """Fire when trailing silence reaches the threshold (classic endpointer)."""
    need = int(math.ceil(threshold_ms / 1000 / frame_s - 1e-9))
    for s, n in runs:
        if n >= need:
            return (s + need) * frame_s
    return None


# ------------------------------------------------------------------ smart-turn
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_STOP_MS = 200   # Silero silence before the model is consulted (Pipecat-style)


@functools.lru_cache(maxsize=1)
def _smart_turn():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor

    path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE)
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess, WhisperFeatureExtractor(chunk_length=8)


def smart_turn_prob(x16: np.ndarray) -> float:
    """P(turn complete) from Smart Turn v3 for the audio so far (last 8 s), as in its reference inference.py."""
    sess, fe = _smart_turn()
    a = x16[-8 * 16000:]
    feats = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length",
               max_length=8 * 16000, truncation=True, do_normalize=True).input_features
    return float(sess.run(None, {"input_features": feats.astype(np.float32)})[0].reshape(-1)[0])


def smart_turn_fire(cands: list[dict], frame_s: float, threshold: float, fallback_ms: float) -> float | None:
    """cands: per silence run {start, len, p_smart (None if run shorter than stop)}."""
    stop = int(math.ceil(SMART_TURN_STOP_MS / 1000 / frame_s))
    fb = int(math.ceil(fallback_ms / 1000 / frame_s))
    for c in cands:
        if c["len"] >= stop and c.get("p_smart") is not None and c["p_smart"] > threshold:
            return (c["start"] + stop) * frame_s
        if c["len"] >= fb:
            return (c["start"] + fb) * frame_s
    return None


# --------------------------------------------------------------- hazard model
CAND_FRAMES = 5      # features are computed once a silence reaches 160 ms
FIRST_FIRE = 6       # earliest possible fire: 192 ms of silence
MAX_FRAMES = 94      # ~3.0 s: training horizon and hard fallback


def hazard_rows(feat: dict, run_len: int, max_frames: int = MAX_FRAMES):
    """Expand one candidate pause into (features, d_ms) rows for k = FIRST_FIRE..min(len, max)."""
    ks = np.arange(FIRST_FIRE, min(run_len, max_frames) + 1)
    return ks, ks * SILERO_FRAME_S * 1000.0


def hazard_fire(cands: list[dict], key: str, theta: float, frame_s: float = SILERO_FRAME_S,
                fallback_frames: int = MAX_FRAMES) -> float | None:
    """cands: per silence run {start, len, <key>: array of P(end | silence >= d) for k=FIRST_FIRE..}."""
    for c in cands:
        p = c.get(key)
        if p is not None and len(p):
            hit = np.nonzero(p >= theta)[0]
            if len(hit):
                return (c["start"] + FIRST_FIRE + int(hit[0])) * frame_s
        if c["len"] >= fallback_frames:
            return (c["start"] + fallback_frames) * frame_s
    return None
