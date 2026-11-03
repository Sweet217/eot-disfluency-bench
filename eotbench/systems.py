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


def smart_turn_decision(cands: list[dict], frame_s: float, threshold: float,
                        fallback_ms: float) -> tuple[float | None, str | None, dict | None]:
    """Returns (fire time, "model" | "cap", candidate). cands: per silence run {start, len, p_smart}."""
    stop = int(math.ceil(SMART_TURN_STOP_MS / 1000 / frame_s))
    fb = int(math.ceil(fallback_ms / 1000 / frame_s))
    for c in cands:
        if stop < fb and c["len"] >= stop and c.get("p_smart") is not None and c["p_smart"] > threshold:
            return (c["start"] + stop) * frame_s, "model", c
        if c["len"] >= fb:
            return (c["start"] + fb) * frame_s, "cap", c
    return None, None, None


def smart_turn_fire(cands: list[dict], frame_s: float, threshold: float, fallback_ms: float) -> float | None:
    return smart_turn_decision(cands, frame_s, threshold, fallback_ms)[0]


# --------------------------------------------------------------- hazard model
CAND_FRAMES = 5      # features are computed once a silence reaches 160 ms
FIRST_FIRE = 6       # earliest possible fire: 192 ms of silence
MAX_FRAMES = 94      # ~3.0 s: training horizon and hard fallback


def cap_frames(cap_ms: float, frame_s: float = SILERO_FRAME_S) -> int:
    return int(math.ceil(cap_ms / 1000 / frame_s - 1e-9))


def hazard_decision(cands: list[dict], key: str, theta: float, frame_s: float = SILERO_FRAME_S,
                    fallback_frames: int = MAX_FRAMES, hold_frames: int | None = None,
                    hold_theta: float = 0.0) -> tuple[float | None, str | None]:
    """Hazard firing rule with a maximum wait and an optional confident hold.

    Fires at the first frame k (>= FIRST_FIRE) where P(end | silence >= d) >= theta ("model"), or
    at the maximum wait `fallback_frames` ("cap"). With a hold, reaching the maximum wait does not
    fire while the model is confident the speaker is mid-turn (P <= hold_theta); the wait may then
    extend up to `hold_frames`, which is the hard limit. hold_frames=None (or <= fallback_frames)
    gives the plain max-wait rule.
    cands: per silence run {start, len, <key>: P for frames k = FIRST_FIRE..min(len, MAX_FRAMES)}.
    """
    nh = fallback_frames if hold_frames is None else max(hold_frames, fallback_frames)
    for c in cands:
        p = c.get(key)
        if p is not None and len(p):
            ks = np.arange(FIRST_FIRE, FIRST_FIRE + len(p))
            hit = p >= theta
            at_cap = (ks >= fallback_frames) & ~((p <= hold_theta) & (ks < nh))
            ev = np.nonzero(hit | at_cap)[0]
            if len(ev):
                k = int(ks[ev[0]])
                return (c["start"] + k) * frame_s, ("model" if hit[ev[0]] else "cap")
            if c["len"] > ks[-1] and c["len"] >= fallback_frames:  # beyond the model's horizon
                return (c["start"] + max(fallback_frames, int(ks[-1]) + 1)) * frame_s, "cap"
        elif c["len"] >= fallback_frames:
            return (c["start"] + fallback_frames) * frame_s, "cap"
    return None, None


def hazard_fire(cands: list[dict], key: str, theta: float, frame_s: float = SILERO_FRAME_S,
                fallback_frames: int = MAX_FRAMES) -> float | None:
    return hazard_decision(cands, key, theta, frame_s, fallback_frames)[0]
