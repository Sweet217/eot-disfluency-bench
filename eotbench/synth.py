"""Synthetic, controlled end-of-turn items built with macOS `say`.

Every item is one user turn:

    [0.5 s room noise] [part A] [hesitation pause, L ms] [part B] [4 s room noise]

Part A is the turn prefix (optionally followed by a filler such as "um" or
"and then"), synthesised with trailing continuation punctuation; part B is the
rest of the turn. Parts are synthesised separately and concatenated, so the
inserted pause length and the true turn end are known exactly (to the 10 ms
trimming resolution). "Control" items have no inserted pause.

Accents: macOS voices whose locale is not English, reading English text, are
used as a rough *approximation* of non-native accents. They are TTS artefacts,
not recordings of L2 speakers, and are labelled `tts_nonnative_approx`.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from multiprocessing import Pool

import numpy as np

from . import audio
from .sentences import SENTENCES, FILLERS

LEAD_S = 0.5
TAIL_S = 4.0
PAUSE_MS = [300, 500, 800, 1200, 1600, 2000]
SNR_DB = [20, 30, 40]
GT_PAUSE_MIN_S = 0.15  # internal silent runs at least this long are labelled pauses
CLEAN_SIL_DB = -45.0   # relative to the loudest frame of the clean turn

VOICES = {
    "en_native": [
        "Samantha", "Fred", "Ralph", "Kathy", "Junior", "Albert", "Daniel",
        "Karen", "Moira", "Tessa", "Eddy (English (US))", "Flo (English (UK))",
        "Reed (English (US))", "Shelley (English (UK))", "Sandy (English (US))",
        "Rocko (English (UK))",
    ],
    "en_IN": ["Aman", "Rishi", "Tara"],
    "tts_nonnative_approx": [
        "Alice", "Anna", "Amélie", "Jacques", "Thomas", "Mónica", "Paulina",
        "Luciana", "Joana", "Kyoko", "Tingting", "Meijia", "Yuna", "Milena",
        "Zosia", "Yelda", "Lekha", "Linh", "Ioana", "Carmit", "Majed", "Xander",
        "Eddy (Spanish (Mexico))", "Flo (German (Germany))",
    ],
}


def voice_split(voices: dict[str, list[str]] = VOICES) -> dict[str, str]:
    """Deterministic speaker(voice)-disjoint split: every third voice per group is test."""
    out = {}
    for g, vs in voices.items():
        for i, v in enumerate(sorted(vs)):
            out[v] = "test" if i % 3 == 2 else "train"
    return out


def sentence_split(n: int = len(SENTENCES)) -> list[str]:
    """Sentences are also split, so a text model cannot memorise test split points."""
    return ["test" if i % 10 in (2, 5, 8) else "train" for i in range(n)]


def say(text: str, voice: str, sr: int = audio.SR) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "o.wav")
        subprocess.run(["say", "-v", voice, f"--data-format=LEI16@{sr}", "-o", p, text],
                       check=True, capture_output=True)
        x, _ = audio.read(p)
    return x


@dataclass
class SynthSpec:
    id: str
    split: str
    voice: str
    accent_group: str
    sentence_idx: int
    split_kind: str          # "mid_clause" | "clause_boundary"
    has_pause: bool
    pause_ms: int            # requested inserted pause (0 for controls)
    filler: str | None
    snr_db: int
    seed: int


def texts_for(spec: SynthSpec) -> tuple[str, str]:
    s = SENTENCES[spec.sentence_idx]
    sep = "|" if "|" in s else "#"
    pre, rest = [t.strip() for t in s.split(sep)]
    f = spec.filler
    if sep == "|":
        a = pre + (f", {f}," if f else ",")
    else:
        a = pre + "." + (f" {f.capitalize()}," if f else "")
    return a, rest


def control_text(spec: SynthSpec) -> str:
    s = SENTENCES[spec.sentence_idx]
    sep = "|" if "|" in s else "#"
    pre, rest = [t.strip() for t in s.split(sep)]
    f = f" {spec.filler}" if spec.filler else ""
    return pre + (f + " " if sep == "|" else "." + f + " ") + rest


def _spread_words(text: str, t0: float, t1: float) -> list[list]:
    """Oracle word end times, spread over a segment proportionally to characters."""
    words = [w.strip(",.?!").lower() for w in text.split()]
    words = [w for w in words if w]
    lens = np.array([len(w) + 1 for w in words], dtype=float)
    ends = t0 + (t1 - t0) * np.cumsum(lens) / lens.sum()
    return [[w, round(float(e), 3)] for w, e in zip(words, ends)]


def internal_pauses(clean: np.ndarray, sr: int, t_on: float, t_end: float) -> list[list[float]]:
    db = audio.frame_db(clean, sr)
    thr = db.max() + CLEAN_SIL_DB
    sil = db < thr
    out, s = [], None
    for i, v in enumerate(list(sil) + [False]):
        t = i * audio.FRAME_S
        if v and s is None:
            s = t
        elif not v and s is not None:
            if s > t_on and t < t_end and t - s >= GT_PAUSE_MIN_S:
                out.append([round(s, 3), round(t, 3)])
            s = None
    return out


def render(spec: SynthSpec, out_dir: str, telephony: bool) -> dict:
    rng = np.random.default_rng(spec.seed)
    sr = audio.SR
    a_txt, b_txt = texts_for(spec)
    if spec.has_pause:
        a = audio.trim(say(a_txt, spec.voice), sr, CLEAN_SIL_DB)
        b = audio.trim(say(b_txt, spec.voice), sr, CLEAN_SIL_DB)
        pause = np.zeros(int(round(spec.pause_ms / 1000 * sr)), np.float32)
        parts = [a, pause, b]
    else:
        # no inserted pause: speak the turn in one go, without the split-point comma
        a_txt = control_text(spec)
        b_txt = ""
        whole = audio.trim(say(a_txt, spec.voice), sr, CLEAN_SIL_DB)
        a, b, parts = whole, whole[:0], [whole]
    speech = np.concatenate(parts)
    # level normalisation: speech RMS (active frames) to -23 dBFS +- 3 dB
    act = speech[np.abs(speech) > 1e-4]
    rms = np.sqrt(np.mean(act ** 2)) + 1e-9
    target = 10 ** ((-23 + rng.uniform(-3, 3)) / 20)
    speech = speech * (target / rms)
    lead, tail = int(LEAD_S * sr), int(TAIL_S * sr)
    clean = np.concatenate([np.zeros(lead, np.float32), speech, np.zeros(tail, np.float32)])
    noise = audio.pink_noise(len(clean), rng) * target * 10 ** (-spec.snr_db / 20)
    mix = (clean + noise).astype(np.float32)

    t_on = LEAD_S
    t_end = LEAD_S + len(speech) / sr
    rec = asdict(spec)
    rec.update(set="synth", text=(a_txt + " " + b_txt).strip(), t_start=round(t_on, 3), t_end=round(t_end, 3),
               duration=round(len(mix) / sr, 3))
    if spec.has_pause:
        a_end = LEAD_S + len(a) / sr
        b_on = a_end + len(pause) / sr
        rec["inserted_pause"] = [round(a_end, 3), round(b_on, 3)]
        rec["words"] = _spread_words(a_txt, t_on, a_end) + _spread_words(b_txt, b_on, t_end)
    else:
        rec["inserted_pause"] = None
        rec["words"] = _spread_words(a_txt, t_on, t_end)
    rec["pauses"] = internal_pauses(clean, sr, t_on, t_end)
    rec["has_filler"] = spec.filler is not None
    rec["speaker"] = spec.voice

    p16 = os.path.join(out_dir, "wav16", spec.id + ".wav")
    audio.write(p16, mix, sr)
    rec["path"] = p16
    if telephony:
        p8 = os.path.join(out_dir, "wav8", spec.id + ".wav")
        audio.write(p8, audio.telephony(mix, sr), 8000)
        rec["path8"] = p8
    return rec


def make_specs(n_train: int, n_test: int, seed: int = 13) -> list[SynthSpec]:
    rnd = random.Random(seed)
    vsplit = voice_split()
    ssplit = sentence_split()
    specs = []
    for split, n in (("train", n_train), ("test", n_test)):
        groups = {g: [v for v in vs if vsplit[v] == split] for g, vs in VOICES.items()}
        sents = [i for i, s in enumerate(ssplit) if s == split]
        for k in range(n):
            g = rnd.choice(sorted(groups))
            v = rnd.choice(groups[g])
            si = rnd.choice(sents)
            has_pause = rnd.random() < 0.85
            filler = rnd.choice(FILLERS) if rnd.random() < 0.6 else None
            specs.append(SynthSpec(
                id=f"syn_{split}_{k:04d}", split=split, voice=v, accent_group=g, sentence_idx=si,
                split_kind="mid_clause" if "|" in SENTENCES[si] else "clause_boundary",
                has_pause=has_pause, pause_ms=rnd.choice(PAUSE_MS) if has_pause else 0,
                filler=filler, snr_db=rnd.choice(SNR_DB), seed=rnd.randrange(2**31)))
    return specs


def _job(args):
    spec, out_dir = args
    try:
        return render(spec, out_dir, telephony=(spec.split == "test"))
    except subprocess.CalledProcessError as e:  # voice missing on this machine
        return {"id": spec.id, "error": str(e)}


def build(out_dir: str, n_train: int, n_test: int, workers: int = 8) -> list[dict]:
    os.makedirs(os.path.join(out_dir, "wav16"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "wav8"), exist_ok=True)
    specs = make_specs(n_train, n_test)
    with Pool(workers) as pool:
        recs = pool.map(_job, [(s, out_dir) for s in specs], chunksize=4)
    bad = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    if bad:
        print(f"[synth] {len(bad)} items failed (missing voices?)")
    return recs
