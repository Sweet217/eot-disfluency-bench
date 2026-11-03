"""Replay every endpointer on the cached sets and compute metrics.

Latency is end to end: the decision frame plus the measured compute on this
machine (VAD frame cost for every system, Smart Turn's measured inference for its
model-triggered fires, the hazard model's measured evaluation cost; feature
availability inside the hazard model already follows measured ASR/prosody costs).
The "instant_*" models are the round-1 ablation and get no compute added.

Operating points are chosen on out-of-fold training predictions only; the test
split is used once, for the reported numbers. Confidence intervals are 95%
percentile intervals from a cluster bootstrap over conversations (EdAcc) or
voices (synthetic).

Writes results/curves.csv, headline.csv, deltas.csv, slices.csv,
operating_points.json, compute_timing.json, goal.json.
"""
from __future__ import annotations

import glob
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eotbench import hazard, metrics, systems, vad  # noqa: E402

CACHE = "data/cache_v2"
TEST = ["edacc_test_16k", "edacc_test_8k", "synth_test_16k", "synth_test_8k"]
TRAIN = {"synth": "synth_train_16k", "edacc": "edacc_train_16k"}
SIL_T = list(range(200, 2001, 100))
MATCH_T = [500, 800, 1200]
SMART_TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CAPS = [400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000]   # hard maximum wait (ms)
THETAS = [round(float(t), 3) for t in np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99, 0.995]])]
# confident hold: past the max wait, keep waiting (up to HOLD_MS) only while P(end) <= HOLD_THETA
HOLD_FIRE_THETAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
HOLD_CAPS = [400, 500, 600, 700, 800, 900, 1000, 1200]
HOLD_MS = [1500, 2000, 3000]
HOLD_THETA = [0.02, 0.05, 0.1, 0.2]
MAIN_MODELS = ["rt_fusion_asr@combined", "rt_fusion_asr@edacc_only"]
WORKERS = 9
B = 2000  # bootstrap resamples
TOL_MS = 1e-6  # latencies equal up to float rounding count as equal (same VAD frame)


def no_slower(s: dict, base: dict) -> bool:
    return (s["latency_p50_ms"] <= base["latency_p50_ms"] + TOL_MS
            and s["latency_p95_ms"] <= base["latency_p95_ms"] + TOL_MS)
FS = vad.SILERO_FRAME_S


def load(name):
    with open(f"{CACHE}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def domain(name: str) -> str:
    return name.split("_")[0]


def cluster_of(meta) -> str:
    return meta["conversation"] if meta["set"] == "edacc" else meta["voice"]


def measure_decide_cost(model, items, fset, n=300) -> float:
    """Median wall-clock cost of one per-frame hazard evaluation (single row), seconds."""
    rows = []
    for it in items:
        for c in it["cands"]:
            if hazard.usable(c):
                X, _ = hazard.expand(c, fset)
                rows += list(X[:3])
        if len(rows) >= n:
            break
    for r in rows[:20]:
        model.predict_proba(r[None, :])
    ts = []
    for r in rows[:n]:
        t0 = time.perf_counter()
        model.predict_proba(r[None, :])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


class Bench:
    def __init__(self):
        self.models = {}
        for p in sorted(glob.glob("data/models/*.pkl")):
            with open(p, "rb") as f:
                m = pickle.load(f)
            self.models[os.path.basename(p)[:-4]] = m
        self.decide_s = {}

    def models_for(self, var: str) -> list[str]:
        out = []
        for name, m in self.models.items():
            if domain(var) not in m["domains"]:
                continue
            if m["fset"].startswith("instant") and not var.endswith("16k"):
                continue
            out.append(name)
        return out

    def fire(self, it, system, param):
        vad_s = it["timing"]["silero_frame_s"]
        if system == "silence_silero":
            t = systems.silence_fire(it["silero_runs"], FS, param)
            return None if t is None else t + vad_s
        if system == "silence_webrtc":
            t = systems.silence_fire(it["webrtc_runs"], vad.WEBRTC_FRAME_S, param)
            return None if t is None else t + it["timing"]["webrtc_frame_s"]
        if system == "smart_turn_v3":
            th, fb = param
            t, why, c = systems.smart_turn_decision(it["cands"], FS, th, fb)
            if t is None:
                return None
            return t + vad_s + (c["smart_s"] if why == "model" else 0.0)
        name = system[len("hazard:"):]
        th, cap, hold, hold_th = param
        t, why = systems.hazard_decision(it["cands"], "p:" + name, th, fallback_frames=systems.cap_frames(cap),
                                         hold_frames=systems.cap_frames(hold), hold_theta=hold_th)
        if t is None:
            return None
        if self.models[name]["fset"].startswith("instant"):
            return t  # ablation: features (and decisions) treated as instantaneous
        return t + vad_s + (self.decide_s[name] if why == "model" else 0.0)

    def fires(self, items, system, param):
        return [self.fire(it, system, param) for it in items]


def grid(system: str, rule: str = "hold"):
    """Settings swept per system. Hazard settings are (theta, max wait, hold limit, hold theta);
    hold limit == max wait means no hold (the plain max-wait rule)."""
    if system == "smart_turn_v3":
        return [(th, fb) for th in SMART_TH for fb in CAPS]
    simple = [(th, cap, cap, 0.0) for th in THETAS for cap in CAPS]
    if rule == "simple":
        return simple
    return simple + [(th, cap, h, hth) for th in HOLD_FIRE_THETAS for cap in HOLD_CAPS
                     for h in HOLD_MS if h > cap for hth in HOLD_THETA]


_G: dict = {}


def _eval_cfg(cfg):
    system, prm = cfg
    return metrics.summarize(_G["bench"].fires(_G["items"], system, prm), _G["t_end"])


def eval_many(bench, items, cfgs) -> list[dict]:
    """Evaluate many (system, setting) pairs in parallel (fork: predictions are already attached)."""
    _G.update(bench=bench, items=items, t_end=[it["meta"]["t_end"] for it in items])
    with mp.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(_eval_cfg, cfgs, chunksize=16)


def choose(bench, items, system, bases: dict, rule: str = "hold") -> dict:
    """Per baseline T: the lowest cut-off rate whose p50 and p95 latency are <= the Silero T ms
    baseline's, all measured on training data (out-of-fold for the hazard models)."""
    prms = grid(system, rule)
    scored = list(zip(prms, eval_many(bench, items, [(system, p) for p in prms])))
    out = {}
    for T, base in bases.items():
        ok = [(prm, s) for prm, s in scored if no_slower(s, base)]
        if not ok:
            out[str(T)] = None
            continue
        prm, s = min(ok, key=lambda x: (x[1]["cutoff_rate"], x[1]["latency_p95_ms"], x[1]["latency_p50_ms"]))
        out[str(T)] = {"param": list(prm), "train": s}
    return out


def label_for(system: str, prm) -> str:
    if system == "smart_turn_v3":
        return f"p>{prm[0]}, max wait {prm[1]} ms"
    th, cap, hold, hth = prm
    if hold > cap:
        return f"theta={th}, max wait {cap} ms, hold to {hold} ms while P<={hth}"
    return f"theta={th}, max wait {cap} ms, no hold"


def pause_bin(meta):
    m = max([e - s for s, e in meta["pauses"]], default=0.0) * 1000
    for hi, lab in [(150, "a) none"), (500, "b) 150-500"), (1000, "c) 500-1000"), (2000, "d) 1000-2000")]:
        if m < hi:
            return lab
    return "e) >=2000"


def slice_keys(meta):
    k = {"max_pause_ms": pause_bin(meta), "filler": "filler" if meta.get("has_filler") else "no_filler",
         "accent_group": meta["accent_group"]}
    if meta["set"] == "synth":
        k["inserted_pause_ms"] = f"{meta['pause_ms']:04d}" if meta["has_pause"] else "0000 (control)"
        k["split_kind"] = meta["split_kind"]
    return k


def ci_row(fires, t_end, clusters):
    s = metrics.summarize(fires, t_end)
    ci = metrics.cluster_bootstrap(fires, t_end, clusters, B=B)
    row = dict(s)
    for k, (lo, hi) in ci.items():
        row[k + "_lo"], row[k + "_hi"] = lo, hi
    return row


def timing_summary(items):
    def q(v):
        v = np.asarray(v, float)
        return {"n": int(len(v)), "median_ms": float(np.median(v) * 1000), "p95_ms": float(np.percentile(v, 95) * 1000)}
    dec = [d for it in items for d in it["timing"]["decode_s"]]
    sc = [d for it in items for d in it["timing"]["score_s"]]
    pr = [d for it in items for d in it["timing"]["prosody_s"]]
    sm = [d for it in items for d in it["timing"]["smart_s"]]
    sil = [it["timing"]["silero_frame_s"] for it in items]
    web = [it["timing"]["webrtc_frame_s"] for it in items]
    ready, txt_ready, fresh_at_first = [], [], []
    for it in items:
        for c in it["cands"]:
            if not hazard.usable(c):
                continue
            onset_known = (c["start"] + 1) * FS
            vers = [v for v in c["text_versions"] if v["launch"] >= onset_known - 1e-9]
            if vers:
                t_txt = min(v["avail"] for v in vers)
                ready.append(max(t_txt, c["prosody_avail"]) - c["onset"])
                txt_ready.append(t_txt - c["onset"])
                fresh_at_first.append(t_txt <= (c["start"] + systems.FIRST_FIRE) * FS + 1e-9)
    return {"asr_decode_mlx_base_en_5s_window": q(dec), "text_score_lm_rules": q(sc), "prosody_fast": q(pr),
            "smart_turn_v3": q(sm), "silero_per_32ms_frame": q(sil), "webrtc_per_30ms_frame": q(web),
            "features_ready_after_silence_onset": q(ready),
            "pause_transcript_ready_after_silence_onset": q(txt_ready),
            "share_of_pauses_with_pause_transcript_by_192ms": float(np.mean(fresh_at_first)) if fresh_at_first else None,
            "decodes_per_item_mean": float(np.mean([it["n_decodes"] for it in items]))}


def main():
    os.makedirs("results", exist_ok=True)
    bench = Bench()
    train = {d: load(n) for d, n in TRAIN.items()}
    # out-of-fold predictions on training items; measured per-frame decision cost per model
    for name, m in bench.models.items():
        for d in m["domains"]:
            for it in train[d]:
                for j, c in enumerate(it["cands"]):
                    p = m["oof"].get((it["meta"]["id"], j))
                    if p is not None:
                        c["p:" + name] = p
        bench.decide_s[name] = measure_decide_cost(m["model"], train[m["domains"][-1]], m["fset"])

    ops = {}
    for d, items in train.items():
        t_end = [it["meta"]["t_end"] for it in items]
        base = {T: metrics.summarize(bench.fires(items, "silence_silero", T), t_end) for T in MATCH_T}
        ops[d] = {"baseline_train": {str(T): b for T, b in base.items()}, "chosen": {}}
        systems_d = ["smart_turn_v3"] + ["hazard:" + n for n in bench.models_for(f"{d}_x_16k")]
        for sysname in systems_d:
            ops[d]["chosen"][sysname] = choose(bench, items, sysname, base)
            if sysname[7:] in MAIN_MODELS:
                ops[d]["chosen"][sysname + "|simple"] = choose(bench, items, sysname, base, rule="simple")
        print(f"[bench] operating points chosen for {d}", flush=True)
    with open("results/operating_points.json", "w") as f:
        json.dump(ops, f, indent=1)

    curves, heads, deltas, slices, goal = [], [], [], [], {}
    for var in TEST:
        items = load(var)
        d = domain(var)
        names = bench.models_for(var)
        for name in names:
            hazard.predict_cands(bench.models[name]["model"], items, bench.models[name]["fset"], "p:" + name)
        t_end = [it["meta"]["t_end"] for it in items]
        clusters = [cluster_of(it["meta"]) for it in items]
        # full trade-off curves (descriptive)
        configs = ([("silence_silero", T) for T in SIL_T] + [("silence_webrtc", T) for T in SIL_T] +
                   [("smart_turn_v3", p) for p in grid("smart_turn_v3")] +
                   [("hazard:" + n, p) for n in names
                    for p in grid("hazard", "hold" if n in MAIN_MODELS or n.startswith("rt_fusion") else "simple")])
        for (system, prm), s in zip(configs, eval_many(bench, items, configs)):
            curves.append(dict(variant=var, system=system, param=str(prm), **s))
        # headline rows: fixed baselines + train-selected operating points
        rows = ([("silence_silero", T, f"{T} ms", None) for T in MATCH_T] +
                [("silence_webrtc", T, f"{T} ms", None) for T in (500, 800)] +
                [("smart_turn_v3", (0.5, 3000), "defaults: p>0.5, fallback 3000 ms", None)])
        for sysname, by_t in ops[d]["chosen"].items():
            real = sysname.split("|")[0]
            if real != "smart_turn_v3" and real[7:] not in names:
                continue
            for T, ch in by_t.items():
                tag = " (ablation: max wait only, no hold)" if sysname.endswith("|simple") else ""
                if ch is None:
                    rows.append((sysname, None, f"no setting meets Silero {T} ms latency on train{tag}", int(T)))
                else:
                    rows.append((sysname, tuple(ch["param"]),
                                 f"{label_for(real, ch['param'])} [train-matched to Silero {T} ms]{tag}", int(T)))
        base_fires = {T: bench.fires(items, "silence_silero", T) for T in MATCH_T}
        for system, prm, label, T in rows:
            system = system.split("|")[0]
            if prm is None:
                heads.append(dict(variant=var, system=system, config=label, match_T=T, feasible=False))
                continue
            fires = bench.fires(items, system, prm)
            heads.append(dict(variant=var, system=system, config=label, match_T=T, feasible=True,
                              **ci_row(fires, t_end, clusters)))
            if T is not None:
                dl = metrics.paired_delta_bootstrap(fires, base_fires[T], t_end, clusters, B=B)
                deltas.append(dict(variant=var, system=system, config=label, vs=f"silence_silero {T} ms",
                                   **{f"{k}{suf}": v[i] for k, v in dl.items()
                                      for i, suf in enumerate(("_delta", "_lo", "_hi"))}))
            key = system in ("silence_silero", "smart_turn_v3") or (T in (500, 800))
            if key:
                groups = {}
                for it, fi in zip(items, fires):
                    for dim, val in slice_keys(it["meta"]).items():
                        groups.setdefault((dim, val), []).append((fi, it["meta"]["t_end"]))
                for (dim, val), fl in groups.items():
                    s = metrics.summarize([a for a, _ in fl], [b for _, b in fl])
                    slices.append(dict(variant=var, system=system, config=label, slice=dim, value=val, **s))
        # explicitly labelled oracle: best fusion setting picked ON TEST under Silero-500 test latencies
        b500 = metrics.summarize(base_fires[500], t_end)
        orc = None
        prms = grid("hazard")
        for prm, s in zip(prms, eval_many(bench, items, [("hazard:rt_fusion_asr@combined", p) for p in prms])):
            if no_slower(s, b500):
                if orc is None or s["cutoff_rate"] < orc[1]["cutoff_rate"]:
                    orc = (prm, s)
        heads.append(dict(variant=var, system="hazard:rt_fusion_asr@combined",
                          config=("ORACLE: setting picked on test (not a valid comparison)" +
                                  (f": {label_for('hazard', orc[0])}" if orc else ", none feasible")),
                          match_T=500, feasible=orc is not None, oracle=True,
                          **({} if orc is None else ci_row(bench.fires(items, "hazard:rt_fusion_asr@combined", orc[0]),
                                                           t_end, clusters))))
        if var == "edacc_test_16k":
            for sysname in [k for k in ops[d]["chosen"] if k.split("|")[0][7:] in MAIN_MODELS]:
                ch = ops[d]["chosen"][sysname]["500"]
                if ch is None:
                    goal[sysname] = {"met": False, "reason": "no setting met the Silero 500 ms latencies on train"}
                    continue
                fires = bench.fires(items, sysname.split("|")[0], tuple(ch["param"]))
                s = metrics.summarize(fires, t_end)
                met = s["cutoff_rate"] < b500["cutoff_rate"] and no_slower(s, b500)
                dl = metrics.paired_delta_bootstrap(fires, base_fires[500], t_end, clusters, B=B)
                goal[sysname] = {"met": bool(met), "param": ch["param"], "label": label_for("hazard", ch["param"]),
                                 "test": s, "baseline_test": b500, "delta_ci": dl}
        print(f"[bench] {var}: {len(items)} items, {len(set(clusters))} clusters", flush=True)

    pd.DataFrame(curves).to_csv("results/curves.csv", index=False)
    pd.DataFrame(heads).to_csv("results/headline.csv", index=False)
    pd.DataFrame(deltas).to_csv("results/deltas.csv", index=False)
    pd.DataFrame(slices).to_csv("results/slices.csv", index=False)
    with open("results/goal.json", "w") as f:
        json.dump(goal, f, indent=1)
    timing = {"test_serial_1_stream": timing_summary(load("edacc_test_16k") + load("synth_test_16k")),
              "train_3_concurrent_streams": timing_summary(train["edacc"] + train["synth"]),
              "hazard_eval_per_frame_ms": {k: v * 1000 for k, v in bench.decide_s.items()}}
    with open("results/compute_timing.json", "w") as f:
        json.dump(timing, f, indent=1)


if __name__ == "__main__":
    main()
