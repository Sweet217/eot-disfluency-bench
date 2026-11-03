"""Per-item extraction, cached to data/cache_v2/<variant>.pkl.

Pass "rt" (real time, measured): Silero and WebRTC VAD frames, the streaming
ASR schedule (MLX Whisper base.en on a rolling window, see eotbench/streaming.py),
fast prosody, pausing-habit features, oracle-text features and Smart Turn, with
the wall-clock cost of every call measured on this machine. Availability times
of all features follow from those costs.

Pass "instant" (ablation, round-1 pipeline): faster-whisper base.en on the
12 s audio prefix (CPU) and pYIN prosody per candidate pause, treated as if
available instantly. Run after "rt" and attached to the same candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from eotbench import audio, features, streaming, systems, vad  # noqa: E402

FS = vad.SILERO_FRAME_S
WEBRTC_AGGRESSIVENESS = 3   # most aggressive setting; 2 labelled room tone as speech on EdAcc
ASR_WINDOW_S = 5.0          # rolling decode window
ASR_HOP_S = 0.5             # re-decode every 0.5 s of new speech
CACHE = "data/cache_v2"
VARIANTS = [
    # name, manifest, split, sample rate, workers for the rt pass
    ("edacc_test_16k", "data/edacc/manifest.jsonl", "test", 16, 1),
    ("edacc_test_8k", "data/edacc/manifest.jsonl", "test", 8, 1),
    ("synth_test_16k", "data/synth/manifest.jsonl", "test", 16, 1),
    ("synth_test_8k", "data/synth/manifest.jsonl", "test", 8, 1),
    ("edacc_train_16k", "data/edacc/manifest.jsonl", "train", 16, 3),
    ("synth_train_16k", "data/synth/manifest.jsonl", "train", 16, 3),
]


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_audio(rec: dict, srv: int) -> np.ndarray:
    if srv == 8:
        x8, sr = audio.read(rec["path8"])
        return audio.resample(x8, sr, 16000)
    return audio.read(rec["path"])[0]


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def process_rt(args):
    rec, srv = args
    features.lm_p_end.cache_clear()  # per-stream cache only: no reuse of other items' scores
    x = load_audio(rec, srv)
    sr = 16000
    p, c_sil = timed(vad.silero_probs, x)
    speech = vad.hysteresis(p)
    s_runs = vad.silence_runs(speech, FS)
    w, c_web = timed(vad.webrtc_speech, x, WEBRTC_AGGRESSIVENESS)
    w_runs = vad.silence_runs(w, vad.WEBRTC_FRAME_S)

    def decode(t_end):
        seg = x[int(max(0.0, t_end - ASR_WINDOW_S) * sr): int(t_end * sr)]
        return timed(features.mlx_transcribe, seg)

    def score(text):
        return timed(features.text_features, text)

    transcripts = streaming.schedule(speech, FS, vad.armed_index(speech, FS), decode, score, hop_s=ASR_HOP_S)
    stop = math.ceil(systems.SMART_TURN_STOP_MS / 1000 / FS)
    timing = {"silero_frame_s": c_sil / max(1, len(p)), "webrtc_frame_s": c_web / max(1, len(w)),
              "decode_s": [tr.decode_s for tr in transcripts], "score_s": [tr.score_s for tr in transcripts],
              "prosody_s": [], "smart_s": [], "oracle_score_s": []}
    cands = []
    for s, n in s_runs:
        c = dict(start=s, len=n, onset=s * FS, end=(s + n) * FS, is_end=(s + n) * FS > rec["t_end"])
        if n >= stop:
            c["p_smart"], c["smart_s"] = timed(systems.smart_turn_prob, x[: int((s + stop) * FS * sr)])
            timing["smart_s"].append(c["smart_s"])
        if n >= systems.CAND_FRAMES:
            t_c = (s + systems.CAND_FRAMES) * FS
            c["f_prosody_rt"], c_p = timed(features.prosody_features_fast, x, sr, s * FS, rec["t_start"])
            c["f_timing"] = features.timing_features(s_runs, s, FS)
            c["prosody_avail"] = t_c + c_p
            timing["prosody_s"].append(c_p)
            t_first = (s + systems.FIRST_FIRE) * FS
            horizon = (s + min(n, systems.MAX_FRAMES)) * FS
            base = streaming.latest_available(transcripts, t_first)
            later = [tr for tr in transcripts if t_first < tr.avail_t <= horizon]
            c["text_versions"] = [dict(avail=tr.avail_t, feats=tr.feats, text=tr.text, kind=tr.kind,
                                       launch=tr.launch_t) for tr in ([base] if base else []) + later]
            c["oracle_text"] = features.oracle_prefix(rec["words"], t_c)
            c["f_text_oracle"], c_o = timed(features.text_features, c["oracle_text"])
            c["oracle_avail"] = t_c + c_o
            timing["oracle_score_s"].append(c_o)
        cands.append(c)
    meta = {k: v for k, v in rec.items() if k != "words"}
    return dict(meta=meta, silero_runs=s_runs, webrtc_runs=w_runs, cands=cands, timing=timing,
                n_frames=len(p), n_decodes=len(transcripts))


def process_instant(args):
    """Round-1 pipeline features (faster-whisper on CPU, pYIN), attached to existing candidates."""
    it, srv = args
    rec = it["meta"]
    x = load_audio(rec, srv)
    for c in it["cands"]:
        if c["len"] >= systems.CAND_FRAMES:
            t_dec = (c["start"] + systems.CAND_FRAMES) * FS
            c["asr_text_instant"] = features.asr_prefix(x, 16000, t_dec)
            c["f_text_asr_instant"] = features.text_features(c["asr_text_instant"])
            c["f_prosody_instant"] = features.prosody_features(x, 16000, c["start"] * FS, rec["t_start"])
    return it


def _init_worker():
    """Load and warm up every model so that no measured call pays loading or JIT costs."""
    import torch

    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    noise = (0.01 * rng.standard_normal(16000 * 3)).astype(np.float32)
    for _ in range(3):
        vad.silero_probs(noise)
        vad.webrtc_speech(noise, WEBRTC_AGGRESSIVENESS)
        features.mlx_transcribe(noise)
        features.text_features("i would like to book a table for")
        features.prosody_features_fast(noise, 16000, 2.5, 0.5)
        systems.smart_turn_prob(noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["rt", "instant", "all"], default="all")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--instant-workers", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    ctx = mp.get_context("spawn")  # MLX/Metal is not fork-safe
    for name, man, split, srv, workers in VARIANTS:
        if args.only and name not in args.only:
            continue
        out = f"{CACHE}/{name}.pkl"
        if args.which in ("rt", "all") and not os.path.exists(out):
            recs = [r for r in load_manifest(man) if r["split"] == split]
            t = time.time()
            with ctx.Pool(workers, initializer=_init_worker) as pool:
                items = pool.map(process_rt, [(r, srv) for r in recs], chunksize=1)
            with open(out, "wb") as f:
                pickle.dump(items, f)
            print(f"[extract:rt] {name}: {len(items)} items, {sum(i['n_decodes'] for i in items)} decodes, "
                  f"{workers} worker(s), {time.time() - t:.0f}s", flush=True)
        if args.which in ("instant", "all") and srv == 16:
            with open(out, "rb") as f:
                items = pickle.load(f)
            if all("instant_done" in i for i in items):
                continue
            t = time.time()
            with ctx.Pool(args.instant_workers, initializer=_init_worker) as pool:
                items = pool.map(process_instant, [(i, srv) for i in items], chunksize=2)
            for i in items:
                i["instant_done"] = True
            with open(out, "wb") as f:
                pickle.dump(items, f)
            print(f"[extract:instant] {name}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
