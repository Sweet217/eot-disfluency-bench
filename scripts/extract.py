"""Run VADs, Smart Turn and feature extraction once per item; cache to data/cache/*.pkl.

Each candidate pause (a Silero silence run of >= 160 ms after speech) gets:
  * p_smart: Smart Turn v3 probability at 200 ms of silence (Pipecat-style),
  * oracle-text features (ground-truth words ending before the decision time),
  * ASR-text features (faster-whisper base.en on the audio prefix),
  * prosodic features (audio before the pause onset).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from eotbench import audio, features, systems, vad  # noqa: E402

FS = vad.SILERO_FRAME_S
WEBRTC_AGGRESSIVENESS = 3  # most aggressive setting; 2 labelled room tone as speech on EdAcc


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def process(args):
    rec, sr_variant = args
    if sr_variant == 8:
        x8, sr = audio.read(rec["path8"])
        x = audio.resample(x8, sr, 16000)
    else:
        x, _ = audio.read(rec["path"])
    sr = 16000
    p = vad.silero_probs(x)
    s_runs = vad.silence_runs(vad.hysteresis(p), vad.SILERO_FRAME_S)
    w_runs = vad.silence_runs(vad.webrtc_speech(x, WEBRTC_AGGRESSIVENESS), vad.WEBRTC_FRAME_S)
    stop = math.ceil(systems.SMART_TURN_STOP_MS / 1000 / FS)
    cands = []
    timing = {"asr": [], "prosody": [], "smart": []}
    for s, n in s_runs:
        c = dict(start=s, len=n, onset=s * FS, end=(s + n) * FS, is_end=(s + n) * FS > rec["t_end"])
        if n >= stop:
            t0 = time.perf_counter()
            c["p_smart"] = systems.smart_turn_prob(x[: int((s + stop) * FS * sr)])
            timing["smart"].append(time.perf_counter() - t0)
        if n >= systems.CAND_FRAMES:
            t_dec = (s + systems.CAND_FRAMES) * FS
            c["oracle_text"] = features.oracle_prefix(rec["words"], t_dec)
            t0 = time.perf_counter()
            c["asr_text"] = features.asr_prefix(x, sr, t_dec)
            timing["asr"].append(time.perf_counter() - t0)
            c["f_text_oracle"] = features.text_features(c["oracle_text"])
            c["f_text_asr"] = features.text_features(c["asr_text"])
            t0 = time.perf_counter()
            c["f_prosody"] = features.prosody_features(x, sr, s * FS, rec["t_start"])
            timing["prosody"].append(time.perf_counter() - t0)
        cands.append(c)
    meta = {k: v for k, v in rec.items() if k not in ("words",)}
    return dict(meta=meta, silero_runs=s_runs, webrtc_runs=w_runs, cands=cands, timing=timing,
                n_frames=len(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    variants = [
        ("synth_train_16k", "data/synth/manifest.jsonl", "train", 16),
        ("edacc_train_16k", "data/edacc/manifest.jsonl", "train", 16),
        ("synth_test_16k", "data/synth/manifest.jsonl", "test", 16),
        ("synth_test_8k", "data/synth/manifest.jsonl", "test", 8),
        ("edacc_test_16k", "data/edacc/manifest.jsonl", "test", 16),
        ("edacc_test_8k", "data/edacc/manifest.jsonl", "test", 8),
    ]
    os.makedirs("data/cache", exist_ok=True)
    for name, man, split, srv in variants:
        if args.only and name not in args.only:
            continue
        out = f"data/cache/{name}.pkl"
        if os.path.exists(out):
            print(f"[extract] {name} cached")
            continue
        recs = [r for r in load_manifest(man) if r["split"] == split]
        t = time.time()
        with Pool(args.workers) as pool:
            items = pool.map(process, [(r, srv) for r in recs], chunksize=2)
        with open(out, "wb") as f:
            pickle.dump(items, f)
        print(f"[extract] {name}: {len(items)} items, "
              f"{sum(len(i['cands']) for i in items)} silence runs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
