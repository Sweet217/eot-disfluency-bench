"""Discrete-time hazard model for end-of-turn.

For a candidate pause with features x (fixed at the pause's decision time) the
model estimates  P(turn has ended | silence has lasted >= d, x)  for every VAD
frame d. Training rows come from every pause that *survived* to d (survival
framing), so the model learns directly how quickly evidence from silence should
override text and prosody. A detector fires at the first frame where the
probability reaches theta; sweeping theta traces the cut-off/latency curve.
"""
from __future__ import annotations

import numpy as np

from .features import PROSODY_FEATURES, TEXT_FEATURES
from .systems import FIRST_FIRE, MAX_FRAMES
from .vad import SILERO_FRAME_S

FEATURE_SETS = {
    "duration_only": [],
    "prosody": ["prosody"],
    "text_oracle": ["text_oracle"],
    "text_asr": ["text_asr"],
    "fusion_oracle": ["prosody", "text_oracle"],
    "fusion_asr": ["prosody", "text_asr"],
}


def columns(fset: str) -> list[str]:
    cols = []
    for part in FEATURE_SETS[fset]:
        if part == "prosody":
            cols += [f"p_{c}" for c in PROSODY_FEATURES]
        else:
            cols += [f"t_{c}" for c in TEXT_FEATURES]
    return cols + ["d_ms", "log_d"]


def cand_vector(c: dict, fset: str) -> np.ndarray:
    v = []
    for part in FEATURE_SETS[fset]:
        if part == "prosody":
            v += [float(c["f_prosody"][k]) for k in PROSODY_FEATURES]
        else:
            src = c["f_text_oracle"] if part == "text_oracle" else c["f_text_asr"]
            v += [float(src[k]) for k in TEXT_FEATURES]
    return np.array(v, float)


def expand(c: dict, fset: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows for frames k = FIRST_FIRE..min(run length, MAX_FRAMES)."""
    ks = np.arange(FIRST_FIRE, min(c["len"], MAX_FRAMES) + 1)
    d = ks * SILERO_FRAME_S * 1000.0
    base = cand_vector(c, fset)
    X = np.column_stack([np.repeat(base[None, :], len(ks), 0), d, np.log(d)]) if len(ks) else np.zeros((0, len(base) + 2))
    return X, ks


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0)


def fit(items: list[dict], fset: str):
    X, y, g = training_rows(items, fset)
    cols = columns(fset)
    mono = [0] * (len(cols) - 2) + [1, 1]  # P(end) non-decreasing in silence duration
    m = make_model()
    m.set_params(monotonic_cst=mono)
    m.fit(X, y)
    return m


def training_rows(items: list[dict], fset: str):
    Xs, ys, gs = [], [], []
    for it in items:
        for c in it["cands"]:
            if "f_prosody" not in c:
                continue
            X, _ = expand(c, fset)
            if len(X):
                Xs.append(X)
                ys.append(np.full(len(X), int(c["is_end"])))
                gs.append(np.array([it["meta"]["speaker"]] * len(X), dtype=object))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def predict_cands(model, items: list[dict], fset: str, key: str) -> None:
    """Attach P(end | silence >= d) arrays (frames FIRST_FIRE..run end) to each candidate as c[key]."""
    for it in items:
        for c in it["cands"]:
            if "f_prosody" not in c:
                continue
            X, _ = expand(c, fset)
            c[key] = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
