"""Discrete-time hazard model for end-of-turn.

For a candidate pause the model estimates, at every VAD frame d of silence,
    P(turn has ended | silence has lasted >= d, features available at that frame).
Training rows come from every pause that *survived* to d (survival framing), so
the model learns how quickly accumulating silence should override text and
prosody. A detector fires at the first frame where the probability reaches
theta, or when the silence reaches a hard maximum wait (cap).

Real-time feature sets ("rt_*") respect measured availability: prosody becomes
usable when its computation has finished, text only when a streaming-ASR
transcript has been decoded and scored (see eotbench/streaming.py). Missing
features are NaN, which gradient-boosted trees handle natively. The "instant_*"
sets reproduce the round-1 pipeline (12 s faster-whisper prefix + pYIN) as if
every feature were available at once; they are kept only as an ablation.
"""
from __future__ import annotations

import numpy as np

from .features import FAST_PROSODY_FEATURES, PROSODY_FEATURES, TEXT_FEATURES, TIMING_FEATURES
from .systems import FIRST_FIRE, MAX_FRAMES
from .vad import SILERO_FRAME_S

FEATURE_SETS = {
    "rt_duration_only": [],
    "rt_prosody": ["prosody_rt"],
    "rt_text_asr": ["text_rt"],
    "rt_fusion_asr": ["prosody_rt", "text_rt"],
    "rt_fusion_oracle": ["prosody_rt", "text_oracle_rt"],
    "instant_fusion_asr": ["prosody_instant", "text_asr_instant"],
    "instant_fusion_oracle": ["prosody_instant", "text_oracle_instant"],
}

GROUP_COLUMNS = {
    "prosody_rt": [f"p_{c}" for c in FAST_PROSODY_FEATURES] + [f"tm_{c}" for c in TIMING_FEATURES],
    "text_rt": [f"t_{c}" for c in TEXT_FEATURES] + ["t_age_s"],
    "text_oracle_rt": [f"o_{c}" for c in TEXT_FEATURES],
    "prosody_instant": [f"pi_{c}" for c in PROSODY_FEATURES],
    "text_asr_instant": [f"ti_{c}" for c in TEXT_FEATURES],
    "text_oracle_instant": [f"oi_{c}" for c in TEXT_FEATURES],
}


def columns(fset: str) -> list[str]:
    cols = []
    for g in FEATURE_SETS[fset]:
        cols += GROUP_COLUMNS[g]
    return cols + ["d_ms", "log_d"]


def frames(c: dict) -> np.ndarray:
    """Decision frames k = FIRST_FIRE..min(run length, MAX_FRAMES) (silence frames since onset)."""
    return np.arange(FIRST_FIRE, min(c["len"], MAX_FRAMES) + 1)


def _group_block(c: dict, g: str, t: np.ndarray) -> np.ndarray:
    n = len(t)
    width = len(GROUP_COLUMNS[g])
    block = np.full((n, width), np.nan)
    if g == "prosody_rt":
        v = [c["f_prosody_rt"][k] for k in FAST_PROSODY_FEATURES] + [c["f_timing"][k] for k in TIMING_FEATURES]
        block[t >= c["prosody_avail"] - 1e-9] = v
    elif g == "text_rt":
        for ver in sorted(c.get("text_versions", []), key=lambda v: v["avail"]):
            m = t >= ver["avail"] - 1e-9
            block[m, :-1] = [ver["feats"][k] for k in TEXT_FEATURES]
            block[m, -1] = t[m] - ver["launch"]  # age of the newest audio the transcript has heard
    elif g == "text_oracle_rt":
        block[t >= c["oracle_avail"] - 1e-9] = [c["f_text_oracle"][k] for k in TEXT_FEATURES]
    elif g == "prosody_instant":
        block[:] = [c["f_prosody_instant"][k] for k in PROSODY_FEATURES]
    elif g == "text_asr_instant":
        block[:] = [c["f_text_asr_instant"][k] for k in TEXT_FEATURES]
    elif g == "text_oracle_instant":
        block[:] = [c["f_text_oracle"][k] for k in TEXT_FEATURES]
    return block


def expand(c: dict, fset: str) -> tuple[np.ndarray, np.ndarray]:
    ks = frames(c)
    t = (c["start"] + ks) * SILERO_FRAME_S
    d = ks * SILERO_FRAME_S * 1000.0
    blocks = [_group_block(c, g, t) for g in FEATURE_SETS[fset]]
    X = np.column_stack(blocks + [d, np.log(d)]) if len(ks) else np.zeros((0, len(columns(fset))))
    return X, ks


def usable(c: dict) -> bool:
    return "f_prosody_rt" in c


def group_key(meta: dict) -> str:
    """Cross-validation group: conversation (EdAcc) or sentence (synthetic, the text-leakage unit)."""
    return meta["conversation"] if meta["set"] == "edacc" else f"sent{meta['sentence_idx']}"


def training_rows(items: list[dict], fset: str):
    Xs, ys, gs = [], [], []
    for it in items:
        g = group_key(it["meta"])
        for c in it["cands"]:
            if not usable(c):
                continue
            X, _ = expand(c, fset)
            if len(X):
                Xs.append(X)
                ys.append(np.full(len(X), int(c["is_end"])))
                gs.append(np.array([g] * len(X), dtype=object))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def make_model(n_cols: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    mono = [0] * (n_cols - 2) + [1, 1]  # P(end) non-decreasing in silence duration
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0,
                                          monotonic_cst=mono)


def fit(items: list[dict], fset: str):
    X, y, _ = training_rows(items, fset)
    m = make_model(X.shape[1])
    m.fit(X, y)
    return m


def predict_cands(model, items: list[dict], fset: str, key: str) -> None:
    """Attach P(end | silence >= d, available features) for frames FIRST_FIRE.. to each candidate."""
    for it in items:
        rows, owners = [], []
        for c in it["cands"]:
            if usable(c):
                X, _ = expand(c, fset)
                if len(X) == 0:
                    c[key] = np.zeros(0)
                    continue
                rows.append(X)
                owners.append((c, len(X)))
        if not rows:
            continue
        P = model.predict_proba(np.vstack(rows))[:, 1]
        o = 0
        for c, n in owners:
            c[key] = P[o:o + n]
            o += n
