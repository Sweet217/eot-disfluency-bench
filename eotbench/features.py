"""Text-completeness and prosodic features computed at a pause onset.

All features are causal: they use only audio (or transcript) up to the moment
the decision is made.
"""
from __future__ import annotations

import functools
import re

import numpy as np

from . import audio

# --------------------------------------------------------------------------- text
FILLERS = {"um", "uh", "er", "erm", "hmm", "ah", "eh", "mm", "like"}
CONJ = {"and", "but", "or", "so", "because", "cause", "then", "that", "which", "if", "when",
        "while", "although", "since", "unless", "whether", "where", "who"}
PREP = {"to", "for", "of", "in", "on", "at", "with", "from", "about", "into", "by", "during",
        "between", "through", "over", "under", "after", "before", "like"}
DET = {"a", "an", "the", "my", "your", "his", "her", "our", "their", "this", "that", "these",
       "those", "some", "any", "every", "each"}
AUX = {"is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
       "can", "could", "will", "would", "should", "may", "might", "must", "i", "we", "you",
       "they", "he", "she", "it's", "i'm", "i'd", "i'll", "we're", "you're", "not", "very", "really"}
TEXT_FEATURES = ["n_words", "last_filler", "last_conj", "last_prep", "last_det", "last_aux",
                 "last2_and_then", "last2_you_know", "lm_p_end", "lm_logit_end"]


def tokens(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z' ]", " ", text.lower()).split() if w]


@functools.lru_cache(maxsize=1)
def _lm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    torch.set_num_threads(2)
    end_ids = sorted({tok.encode(s)[0] for s in [".", "?", "!", " .", "\n"]})
    return tok, model, end_ids


@functools.lru_cache(maxsize=50000)
def lm_p_end(text: str) -> float:
    """P(next token ends the sentence | lowercased, unpunctuated prefix) under distilgpt2."""
    import torch

    tok, model, end_ids = _lm()
    t = " ".join(tokens(text))
    if not t:
        return 0.0
    ids = tok.encode(t[0].upper() + t[1:])[-64:]
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1]
    p = torch.softmax(logits.float(), -1)
    return float(p[end_ids].sum())


def text_features(text: str) -> dict:
    w = tokens(text)
    last = w[-1] if w else ""
    last2 = " ".join(w[-2:])
    p = lm_p_end(text) if w else 0.0
    return dict(n_words=len(w), last_filler=float(last in FILLERS), last_conj=float(last in CONJ),
                last_prep=float(last in PREP), last_det=float(last in DET), last_aux=float(last in AUX),
                last2_and_then=float(last2 == "and then"), last2_you_know=float(last2 == "you know"),
                lm_p_end=p, lm_logit_end=float(np.log(p + 1e-6) - np.log(1 - p + 1e-6)))


def oracle_prefix(words: list[list], t: float) -> str:
    """Ground-truth words whose (estimated) end time is <= t."""
    return " ".join(w for w, e in words if e <= t + 1e-3)


# ------------------------------------------------------------------------- prosody
PROSODY_FEATURES = ["f0_slope_st_s", "f0_last_rel_st", "f0_voiced_frac", "energy_slope_db_s",
                    "energy_last_rel_db", "final_voiced_run_s", "final_lengthening", "turn_dur_s"]


def prosody_features(x: np.ndarray, sr: int, t_pause: float, t_turn_start: float) -> dict:
    """Prosodic cues just before the pause onset t_pause (seconds), using <= 2 s of history."""
    import librosa

    t0 = max(0.0, t_pause - 2.0)
    seg = x[int(t0 * sr): int(t_pause * sr)]
    out = dict.fromkeys(PROSODY_FEATURES, 0.0)
    out["turn_dur_s"] = max(0.0, t_pause - t_turn_start)
    if len(seg) < int(0.3 * sr):
        return out
    hop = int(0.01 * sr)
    f0, vflag, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024 if sr == 16000 else 512,
                                hop_length=hop, center=False)
    voiced = np.where(vflag & np.isfinite(f0))[0]
    db = audio.frame_db(seg, sr)
    if len(voiced) >= 5:
        st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        tv = voiced * 0.01
        last = voiced[-1]
        win = voiced >= last - 30  # last ~300 ms of voiced frames
        if win.sum() >= 4:
            out["f0_slope_st_s"] = float(np.polyfit(tv[win], st[win], 1)[0])
        out["f0_last_rel_st"] = float(np.mean(st[-3:]))
        out["f0_voiced_frac"] = float(np.mean(vflag[-30:]))
        # final voiced run vs typical voiced run in the window (final lengthening proxy)
        runs, s = [], voiced[0]
        for a, b in zip(voiced[:-1], voiced[1:]):
            if b != a + 1:
                runs.append((a - s + 1) * 0.01)
                s = b
        runs.append((voiced[-1] - s + 1) * 0.01)
        out["final_voiced_run_s"] = runs[-1]
        out["final_lengthening"] = float(runs[-1] / (np.median(runs) + 1e-3))
    if len(db) >= 20:
        tail = db[-20:]
        out["energy_slope_db_s"] = float(np.polyfit(np.arange(20) * 0.01, tail, 1)[0])
        out["energy_last_rel_db"] = float(np.mean(db[-5:]) - np.percentile(db, 90))
    return out


# ----------------------------------------------------------------------------- ASR
@functools.lru_cache(maxsize=2)
def whisper(name: str = "base.en"):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2)


def asr_prefix(x: np.ndarray, sr: int, t: float, model: str = "base.en", max_s: float = 12.0) -> str:
    """Transcribe the audio prefix [t - max_s, t] (a stand-in for a streaming ASR hypothesis)."""
    seg = x[int(max(0.0, t - max_s) * sr): int(t * sr)]
    if sr != 16000:
        seg = audio.resample(seg, sr, 16000)
    if len(seg) < 1600:
        return ""
    segs, _ = whisper(model).transcribe(seg, language="en", beam_size=1, vad_filter=False,
                                        condition_on_previous_text=False, without_timestamps=True)
    return " ".join(s.text for s in segs).strip()


def asr_words(x: np.ndarray, model: str = "small.en") -> list[tuple[str, float]]:
    segs, _ = whisper(model).transcribe(x, language="en", beam_size=5, word_timestamps=True,
                                        vad_filter=False)
    return [(w.word.strip(), float(w.end)) for s in segs for w in (s.words or [])]


# ------------------------------------------------------- real-time (streaming) path
FAST_PROSODY_FEATURES = ["f0_slope_st_s", "f0_last_rel_st", "f0_voiced_frac", "energy_slope_db_s",
                         "energy_last_rel_db", "final_voiced_run_s", "final_lengthening", "turn_dur_s"]
TIMING_FEATURES = ["n_prev_pauses", "mean_prev_pause_s", "max_prev_pause_s", "since_prev_pause_s"]


def prosody_features_fast(x: np.ndarray, sr: int, t_pause: float, t_turn_start: float,
                          window_s: float = 1.0) -> dict:
    """Cheap prosody for real-time use: YIN F0 on the last `window_s` before the pause onset.

    Same feature names as `prosody_features`, but voicing is decided from energy and
    zero-crossing rate instead of pYIN's HMM, so it runs in ~2 ms instead of ~250 ms.
    """
    import librosa

    out = dict.fromkeys(FAST_PROSODY_FEATURES, 0.0)
    out["turn_dur_s"] = max(0.0, t_pause - t_turn_start)
    seg = x[int(max(0.0, t_pause - window_s) * sr): int(t_pause * sr)]
    hop, flen = int(0.01 * sr), int(0.064 * sr)
    if len(seg) < flen + 10 * hop:
        return out
    f0 = librosa.yin(seg, fmin=60, fmax=400, sr=sr, frame_length=flen, hop_length=hop, center=False)
    n = len(f0)
    frames = np.lib.stride_tricks.sliding_window_view(seg, flen)[::hop][:n]
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    vflag = (rms_db > rms_db.max() - 25) & (zcr < 0.25) & (f0 > 62) & (f0 < 395)
    voiced = np.where(vflag)[0]
    if len(voiced) >= 5:
        st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        last = voiced[-1]
        win = voiced >= last - 30
        if win.sum() >= 4:
            out["f0_slope_st_s"] = float(np.polyfit(voiced[win] * 0.01, st[win], 1)[0])
        out["f0_last_rel_st"] = float(np.mean(st[-3:]))
        out["f0_voiced_frac"] = float(np.mean(vflag[-30:]))
        runs, s = [], voiced[0]
        for a, b in zip(voiced[:-1], voiced[1:]):
            if b != a + 1:
                runs.append((a - s + 1) * 0.01)
                s = b
        runs.append((voiced[-1] - s + 1) * 0.01)
        out["final_voiced_run_s"] = float(runs[-1])
        out["final_lengthening"] = float(runs[-1] / (np.median(runs) + 1e-3))
    db = audio.frame_db(seg, sr)
    if len(db) >= 20:
        out["energy_slope_db_s"] = float(np.polyfit(np.arange(20) * 0.01, db[-20:], 1)[0])
        out["energy_last_rel_db"] = float(np.mean(db[-5:]) - np.percentile(db, 90))
    return out


def timing_features(prev_runs: list[tuple[int, int]], start: int, frame_s: float,
                    min_frames: int = 5) -> dict:
    """Pausing habit so far in this turn: earlier silences of >= min_frames before `start`."""
    prev = [(s, n) for s, n in prev_runs if s < start and n >= min_frames]
    lens = [n * frame_s for _, n in prev]
    last_end = (prev[-1][0] + prev[-1][1]) if prev else None
    return dict(n_prev_pauses=float(len(prev)),
                mean_prev_pause_s=float(np.mean(lens)) if lens else 0.0,
                max_prev_pause_s=float(max(lens)) if lens else 0.0,
                since_prev_pause_s=float((start - last_end) * frame_s) if last_end is not None else -1.0)


MLX_ASR_MODEL = "mlx-community/whisper-base.en-mlx"
MLX_ALIGN_MODEL = "mlx-community/whisper-small.en-mlx"


def mlx_transcribe(seg: np.ndarray, model: str = MLX_ASR_MODEL) -> str:
    """Greedy, timestamp-free decode of a short window with MLX Whisper (Apple-silicon GPU)."""
    import mlx_whisper

    if len(seg) < 1600:
        return ""
    out = mlx_whisper.transcribe(seg.astype(np.float32), path_or_hf_repo=model, language="en",
                                 temperature=0.0, condition_on_previous_text=False,
                                 without_timestamps=True, verbose=None)
    return out["text"].strip()


def mlx_words(x: np.ndarray, model: str = MLX_ALIGN_MODEL) -> list[tuple[str, float]]:
    import mlx_whisper

    out = mlx_whisper.transcribe(x.astype(np.float32), path_or_hf_repo=model, language="en",
                                 word_timestamps=True, temperature=0.0, verbose=None)
    return [(w["word"].strip(), float(w["end"])) for s in out["segments"] for w in s.get("words", [])]
