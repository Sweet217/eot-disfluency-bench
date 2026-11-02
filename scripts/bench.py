"""Replay every endpointer on the cached test sets and compute metrics.

Writes results/curves.csv (all thresholds), results/headline.csv,
results/slices.csv, results/operating_points.json and results/compute_timing.json.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eotbench import hazard, metrics, systems, vad  # noqa: E402

TEST = ["synth_test_16k", "synth_test_8k", "edacc_test_16k", "edacc_test_8k"]
TRAIN = ["synth_train_16k", "edacc_train_16k"]
SIL_T = list(range(200, 2001, 100))
HEADLINE_T = [300, 500, 800, 1200]
SMART_TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CAPS = [600, 800, 1000, 1200, 1500, 2000, 3000]   # silence cap (fallback) in ms
SMART_FB = CAPS
THETAS = [round(float(t), 3) for t in np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99, 0.995]])]
MATCH_T = [500, 800, 1200]
HEADLINE_FSETS = ["fusion_asr", "fusion_oracle", "prosody", "text_asr", "text_oracle", "duration_only"]


def load(name):
    with open(f"data/cache/{name}.pkl", "rb") as f:
        return pickle.load(f)


def fires_for(items, system, param):
    out = []
    for it in items:
        if system == "silence_silero":
            out.append(systems.silence_fire(it["silero_runs"], vad.SILERO_FRAME_S, param))
        elif system == "silence_webrtc":
            out.append(systems.silence_fire(it["webrtc_runs"], vad.WEBRTC_FRAME_S, param))
        elif system == "smart_turn_v3":
            th, fb = param
            out.append(systems.smart_turn_fire(it["cands"], vad.SILERO_FRAME_S, th, fb))
        elif system.startswith("hazard_"):
            th, cap = param
            out.append(systems.hazard_fire(it["cands"], "p_" + system[7:], th,
                                           fallback_frames=int(math.ceil(cap / 1000 / vad.SILERO_FRAME_S))))
    return out


def pause_bin(meta):
    m = max([e - s for s, e in meta["pauses"]], default=0.0) * 1000
    if m < 150:
        return "a) none"
    for lo, hi, lab in [(150, 500, "b) 150-500"), (500, 1000, "c) 500-1000"), (1000, 2000, "d) 1000-2000")]:
        if m < hi:
            return lab
    return "e) >=2000"


def slice_keys(meta):
    k = {"all": "all", "max_pause_ms": pause_bin(meta), "filler": "filler" if meta.get("has_filler") else "no_filler",
         "accent_group": meta["accent_group"]}
    if meta["set"] == "synth":
        k["inserted_pause_ms"] = str(meta["pause_ms"]) if meta["has_pause"] else "0 (control)"
        k["split_kind"] = meta["split_kind"]
        k["filler_type"] = meta["filler"] or "none"
    return k


def domain(name: str) -> str:
    return name.split("_")[0]


def tunable_configs(fset_or_smart: str):
    if fset_or_smart == "smart_turn_v3":
        return [("smart_turn_v3", (th, fb)) for th in SMART_TH for fb in SMART_FB]
    return [("hazard_" + fset_or_smart, (th, cap)) for th in THETAS for cap in CAPS]


def choose_operating_points(train_by_domain):
    """Per domain, on training data only (out-of-fold for the hazard models):
    pick the config with the lowest cut-off rate whose p50 AND p95 latency are both
    <= those of the Silero silence baseline at T ms on the same training items."""
    ops = {}
    for dom, items in train_by_domain.items():
        t_end = [it["meta"]["t_end"] for it in items]
        ops[dom] = {"baseline_train": {}, "chosen": {}}
        for T in MATCH_T:
            ops[dom]["baseline_train"][str(T)] = metrics.summarize(fires_for(items, "silence_silero", T), t_end)
        for name in ["smart_turn_v3"] + list(hazard.FEATURE_SETS):
            ops[dom]["chosen"][name] = {}
            scored = [(sys_, prm, metrics.summarize(fires_for(items, sys_, prm), t_end))
                      for sys_, prm in tunable_configs(name)]
            for T in MATCH_T:
                b = ops[dom]["baseline_train"][str(T)]
                ok = [x for x in scored if x[2]["latency_p50_ms"] <= b["latency_p50_ms"]
                      and x[2]["latency_p95_ms"] <= b["latency_p95_ms"]]
                if ok:
                    best = min(ok, key=lambda x: (x[2]["cutoff_rate"], x[2]["latency_p50_ms"]))
                    ops[dom]["chosen"][name][str(T)] = {"param": list(best[1]), "train": best[2], "feasible": True}
                else:
                    ops[dom]["chosen"][name][str(T)] = {"param": None, "feasible": False}
    return ops


def attach_oof(items):
    for fset in hazard.FEATURE_SETS:
        with open(f"data/models/{fset}.pkl", "rb") as f:
            oof = pickle.load(f)["oof"]
        for it in items:
            for j, c in enumerate(it["cands"]):
                c["p_" + fset] = oof.get((it["meta"]["id"], j))


def main():
    os.makedirs("results", exist_ok=True)
    train_by_domain = {}
    for n in TRAIN:
        items = load(n)
        attach_oof(items)
        train_by_domain[domain(n)] = items
    ops = choose_operating_points(train_by_domain)
    with open("results/operating_points.json", "w") as f:
        json.dump(ops, f, indent=2)
    models = {}
    for fset in hazard.FEATURE_SETS:
        with open(f"data/models/{fset}.pkl", "rb") as f:
            models[fset] = pickle.load(f)["model"]

    curves, heads, slices = [], [], []
    timing = {"asr_s": [], "prosody_s": [], "smart_s": []}
    for var in TEST:
        items = load(var)
        dom_ops = ops[domain(var)]["chosen"]
        for fset, m in models.items():
            hazard.predict_cands(m, items, fset, "p_" + fset)
        if var.endswith("16k"):
            for it in items:
                timing["asr_s"] += it["timing"]["asr"]
                timing["prosody_s"] += it["timing"]["prosody"]
                timing["smart_s"] += it["timing"]["smart"]
        t_end = [it["meta"]["t_end"] for it in items]
        configs = ([("silence_silero", T, f"{T} ms") for T in SIL_T] +
                   [("silence_webrtc", T, f"{T} ms") for T in SIL_T] +
                   [(sy, prm, f"p>{prm[0]}, fallback {prm[1]} ms") for sy, prm in tunable_configs("smart_turn_v3")] +
                   [(sy, prm, f"theta={prm[0]}, cap {prm[1]} ms")
                    for fs in hazard.FEATURE_SETS for sy, prm in tunable_configs(fs)])
        head_cfg = ([("silence_silero", T, f"{T} ms") for T in HEADLINE_T] +
                    [("silence_webrtc", T, f"{T} ms") for T in HEADLINE_T] +
                    [("smart_turn_v3", (0.5, 3000), "p>0.5, fallback 3000 ms (defaults)")])
        for T in MATCH_T:
            for name in ["smart_turn_v3"] + HEADLINE_FSETS:
                ch = dom_ops[name][str(T)]
                if not ch["feasible"]:
                    continue
                prm = tuple(ch["param"])
                sy = name if name == "smart_turn_v3" else "hazard_" + name
                knob = "p>" if name == "smart_turn_v3" else "theta="
                head_cfg.append((sy, prm, f"{knob}{prm[0]}, cap {prm[1]} ms [matched to Silero {T} ms on train]"))
        for system, param, label in configs:
            s = metrics.summarize(fires_for(items, system, param), t_end)
            curves.append(dict(variant=var, system=system, param=str(param), label=label, **s))
        for system, param, label in head_cfg:
            fires = fires_for(items, system, param)
            heads.append(dict(variant=var, system=system, config=label,
                              **metrics.summarize(fires, t_end)))
            groups = {}
            for it, fi in zip(items, fires):
                for dim, val in slice_keys(it["meta"]).items():
                    groups.setdefault((dim, val), []).append((fi, it["meta"]["t_end"]))
            for (dim, val), fl in groups.items():
                s = metrics.summarize([a for a, _ in fl], [b for _, b in fl])
                slices.append(dict(variant=var, system=system, config=label, slice=dim, value=val, **s))
        print(f"[bench] {var}: {len(items)} items")
    pd.DataFrame(curves).to_csv("results/curves.csv", index=False)
    pd.DataFrame(heads).to_csv("results/headline.csv", index=False)
    pd.DataFrame(slices).to_csv("results/slices.csv", index=False)
    with open("results/compute_timing.json", "w") as f:
        json.dump({k: {"n": len(v), "median_ms": float(np.median(v) * 1000), "p95_ms": float(np.percentile(v, 95) * 1000)}
                   for k, v in timing.items() if v}, f, indent=2)


if __name__ == "__main__":
    main()
