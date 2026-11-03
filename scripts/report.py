"""Make Pareto plots, results/RESULTS.md and the README result tables from results/*.

README tables live between markers such as <!-- results:edacc_test_16k --> and
<!-- /results:edacc_test_16k --> and are regenerated here, so every number in the
README comes from the CSVs written by bench.py.
"""
from __future__ import annotations

import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VARIANTS = {
    "edacc_test_16k": "EdAcc real conversation, 16 kHz",
    "edacc_test_8k": "EdAcc real conversation, 8 kHz telephony",
    "synth_test_16k": "Synthetic TTS, 16 kHz",
    "synth_test_8k": "Synthetic TTS, 8 kHz telephony",
}
CURVES = [
    ("silence_silero", "Silence threshold (Silero VAD)", "#7f7f7f", "o", "-"),
    ("silence_webrtc", "Silence threshold (WebRTC VAD)", "#bcbcbc", "s", "-"),
    ("smart_turn_v3", "Smart Turn v3 (threshold x max wait)", "#d62728", "^", "-"),
    ("hazard:rt_prosody@combined", "Hazard: prosody only", "#9467bd", ".", "-"),
    ("hazard:rt_text_asr@combined", "Hazard: streaming-ASR text only", "#2ca02c", ".", "-"),
    ("hazard:rt_fusion_asr@combined", "Hazard: prosody + streaming ASR", "#1f77b4", "D", "-"),
    ("hazard:rt_fusion_asr@edacc_only", "Hazard: prosody + streaming ASR (EdAcc-only training)", "#ff7f0e", "D", "-"),
    ("hazard:rt_fusion_oracle@combined", "Hazard: prosody + oracle text", "#17becf", "x", "-"),
    ("hazard:instant_fusion_asr@combined", "Ablation: round-1 features, instant (no compute)", "#1f77b4", None, ":"),
]
NICE = {
    "silence_silero": "Silence (Silero VAD)",
    "silence_webrtc": "Silence (WebRTC VAD)",
    "smart_turn_v3": "Smart Turn v3",
    "hazard:rt_fusion_asr@combined": "**Hazard: prosody + streaming ASR**",
    "hazard:rt_fusion_asr@edacc_only": "**Hazard: prosody + streaming ASR, EdAcc-only training**",
    "hazard:rt_fusion_oracle@combined": "Hazard: prosody + oracle text",
    "hazard:rt_prosody@combined": "Hazard: prosody only",
    "hazard:rt_text_asr@combined": "Hazard: streaming-ASR text only",
    "hazard:rt_duration_only@combined": "Hazard: silence duration only",
    "hazard:instant_fusion_asr@combined": "Ablation: round-1 features, instant",
    "hazard:instant_fusion_oracle@combined": "Ablation: round-1 features + oracle text, instant",
}


def nice(system: str) -> str:
    return NICE.get(system, system.replace("hazard:", "Hazard: "))


def fms_cmp(a, b):
    """Format two latencies; show 3 decimals when they differ by less than 1 ms."""
    if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in (a, b)) or abs(a - b) >= 1:
        return fms(a), fms(b)
    return f"{a:.3f}", f"{b:.3f}"


def fms(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return "inf" if math.isinf(v) else f"{v:.0f}"


def fpct(v):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f}%"


def cell(r, k, pct):
    f = fpct if pct else fms
    if not r.get("feasible", True) or pd.isna(r.get(k)):
        return "-"
    lo, hi = r.get(k + "_lo"), r.get(k + "_hi")
    if lo is None or pd.isna(lo):
        return f(r[k])
    lo_s, hi_s = (f"{100 * lo:.1f}", f"{100 * hi:.1f}") if pct else (fms(lo), fms(hi))
    return f"{f(r[k])} ({lo_s}-{hi_s})"


def frontier(df: pd.DataFrame, lat: str) -> pd.DataFrame:
    d = df[np.isfinite(df[lat])].sort_values([lat, "cutoff_rate"])
    keep, best = [], np.inf
    for i, r in d.iterrows():
        if r.cutoff_rate < best - 1e-12:
            keep.append(i)
            best = r.cutoff_rate
    return d.loc[keep]


def pareto_plot(curves, heads, var, out, lat="latency_p50_ms"):
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150)
    d = curves[curves.variant == var]
    for system, label, color, marker, ls in CURVES:
        c = frontier(d[d.system == system], lat)
        if c.empty:
            continue
        step = system.startswith(("smart", "hazard"))
        ax.plot(c[lat], 100 * c.cutoff_rate, marker=marker, ms=3, lw=1.3, ls=ls, color=color, label=label,
                drawstyle="steps-post" if step else "default")
    h = heads[(heads.variant == var) & (heads.feasible == True) & (heads.get("oracle") != True)]  # noqa: E712
    stars = {"hazard:rt_fusion_asr@combined": ("train-picked (combined)", (8, 4)),
             "hazard:rt_fusion_asr@edacc_only": ("train-picked (EdAcc-only)", (8, -12))}
    for _, r in h.iterrows():
        if r.system in stars and r.match_T == 500 and "ablation" not in str(r.config):
            txt, off = stars[r.system]
            ax.plot(r[lat], 100 * r.cutoff_rate, marker="*", ms=11, color="black", zorder=5)
            ax.annotate(txt, (r[lat], 100 * r.cutoff_rate), textcoords="offset points", xytext=off, fontsize=7)
    for T in (300, 500, 800, 1200):
        r = d[(d.system == "silence_silero") & (d.param == str(T))]
        if len(r) and np.isfinite(r[lat].iloc[0]):
            ax.annotate(f"{T} ms", (r[lat].iloc[0], 100 * r.cutoff_rate.iloc[0]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="#555")
    which = "median (p50)" if lat.endswith("p50_ms") else "p95"
    ax.set_xlabel(f"End-to-end end-of-turn latency incl. measured compute, {which} (ms)")
    ax.set_ylabel("Premature cut-off rate (% of turns)")
    ax.set_title(f"{VARIANTS[var]}: Pareto frontier per system (test set)", fontsize=10)
    ax.set_xlim(0, 3300)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def headline_table(h: pd.DataFrame, compact: bool) -> list[str]:
    L = ["| system | setting | cut-off | p50 latency, ms | p95 latency, ms | late/miss |",
         "|---|---|---:|---:|---:|---:|"]
    for _, r in h.iterrows():
        if compact:
            if r.system == "silence_webrtc" and r.config != "800 ms":
                continue
            if r.system.startswith("hazard:") and r.system not in (
                    "hazard:rt_fusion_asr@combined", "hazard:rt_fusion_asr@edacc_only",
                    "hazard:rt_fusion_oracle@combined", "hazard:instant_fusion_asr@combined"):
                continue
            if r.system in ("hazard:rt_fusion_oracle@combined", "hazard:instant_fusion_asr@combined") and r.match_T != 500:
                continue
            if r.system == "smart_turn_v3" and r.match_T == 1200:
                continue
            if "ablation" in str(r.config) and (r.match_T != 500 or r.system != "hazard:rt_fusion_asr@combined"):
                continue
            if r.system == "hazard:rt_fusion_asr@edacc_only" and r.match_T == 1200:
                continue
        L.append(f"| {nice(r.system)} | {r.config} | {cell(r, 'cutoff_rate', True)} | "
                 f"{cell(r, 'latency_p50_ms', False)} | {cell(r, 'latency_p95_ms', False)} | "
                 f"{cell(r, 'late_or_miss_rate', True)} |")
    return L


def delta_table(dl: pd.DataFrame, var: str, compact: bool) -> list[str]:
    d = dl[dl.variant == var]
    if compact:
        d = d[d.system.isin(["hazard:rt_fusion_asr@combined", "hazard:rt_fusion_asr@edacc_only", "smart_turn_v3"])
              & ~d.config.str.contains("ablation")]
    L = ["| system | setting | vs | Δ cut-off (pp) | Δ p50 (ms) | Δ p95 (ms) |", "|---|---|---|---:|---:|---:|"]

    def c(r, k, pct):
        sc = 100 if pct else 1
        v, lo, hi = r[k + "_delta"] * sc, r[k + "_lo"] * sc, r[k + "_hi"] * sc
        if any(map(lambda z: isinstance(z, float) and (math.isnan(z) or math.isinf(z)), (v, lo, hi))):
            return "-" if math.isnan(v) else f"{v:+.1f}" if pct else f"{v:+.0f}"
        return f"{v:+.1f} ({lo:+.1f} to {hi:+.1f})" if pct else f"{v:+.0f} ({lo:+.0f} to {hi:+.0f})"
    for _, r in d.iterrows():
        L.append(f"| {nice(r.system)} | {r.config.split(' [')[0]} | {r.vs.replace('silence_silero', 'Silero')} | "
                 f"{c(r, 'cutoff_rate', True)} | {c(r, 'latency_p50_ms', False)} | {c(r, 'latency_p95_ms', False)} |")
    return L


def compute_table(timing: dict) -> list[str]:
    t = timing["test_serial_1_stream"]
    tr = timing["train_3_concurrent_streams"]
    rows = [("Streaming ASR decode, MLX Whisper base.en, 5 s window (GPU)", "asr_decode_mlx_base_en_5s_window"),
            ("Text score: distilgpt2 end-of-sentence probability + rules (CPU)", "text_score_lm_rules"),
            ("Fast prosody: YIN F0, energy, lengthening (CPU)", "prosody_fast"),
            ("Smart Turn v3 inference incl. features (CPU)", "smart_turn_v3"),
            ("Silero VAD, per 32 ms frame", "silero_per_32ms_frame"),
            ("WebRTC VAD, per 30 ms frame", "webrtc_per_30ms_frame"),
            ("Transcript covering the pause ready, after silence onset", "pause_transcript_ready_after_silence_onset"),
            ("All hazard features ready, after silence onset (prosody runs at 160 ms)",
             "features_ready_after_silence_onset")]
    L = ["| component | median ms (1 stream) | p95 ms (1 stream) | median ms (3 concurrent streams) | calls |",
         "|---|---:|---:|---:|---:|"]
    for lab, k in rows:
        L.append(f"| {lab} | {t[k]['median_ms']:.1f} | {t[k]['p95_ms']:.1f} | {tr[k]['median_ms']:.1f} | {t[k]['n']} |")
    he = timing["hazard_eval_per_frame_ms"]
    L.append(f"| Hazard model evaluation, per 32 ms frame (CPU) | "
             f"{np.median([v for k, v in he.items() if k.startswith('rt_')]):.2f} | - | - | - |")
    L += ["", f"Share of candidate pauses whose post-pause transcript was ready by the earliest decision "
              f"(192 ms of silence): {100 * t['share_of_pauses_with_pause_transcript_by_192ms']:.1f}%. "
              f"Mean decodes per test turn: {t['decodes_per_item_mean']:.1f}."]
    return L


def replace_block(text: str, key: str, lines: list[str]) -> str:
    pat = re.compile(rf"(<!-- results:{key} -->).*?(<!-- /results:{key} -->)", re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: m.group(1) + "\n" + "\n".join(lines) + "\n" + m.group(2), text)


def main():
    curves = pd.read_csv("results/curves.csv")
    heads = pd.read_csv("results/headline.csv")
    if "oracle" not in heads:
        heads["oracle"] = False
    heads["oracle"] = heads["oracle"].fillna(False).astype(bool)
    dl = pd.read_csv("results/deltas.csv")
    slices = pd.read_csv("results/slices.csv")
    goal = json.load(open("results/goal.json"))
    timing = json.load(open("results/compute_timing.json"))
    ops = json.load(open("results/operating_points.json"))
    os.makedirs("results/plots", exist_ok=True)
    for var in VARIANTS:
        pareto_plot(curves, heads, var, f"results/plots/pareto_{var}.png")
        pareto_plot(curves, heads, var, f"results/plots/pareto_p95_{var}.png", lat="latency_p95_ms")

    L = ["# Results", "", "Generated by `scripts/report.py` from `results/*.csv`; every number comes from runs of",
         "this repository on an Apple M1 Max. Latency is **end to end**: decision time plus measured compute.",
         "Cells show the point estimate and a 95% cluster-bootstrap interval (2000 resamples of conversations",
         "for EdAcc, of voices for the synthetic set). Every operating point was chosen on training data.", ""]
    L += ["## Goal check (EdAcc 16 kHz, train-chosen operating point vs. Silero 500 ms)", "",
          "Goal: fewer cut-offs than the 500 ms silence baseline with p50 and p95 latency no higher.", ""]
    for k, g in goal.items():
        if "test" in g:
            a50, b50 = fms_cmp(g["test"]["latency_p50_ms"], g["baseline_test"]["latency_p50_ms"])
            a95, b95 = fms_cmp(g["test"]["latency_p95_ms"], g["baseline_test"]["latency_p95_ms"])
            L.append(f"* {nice(k.split('|')[0])}{' (ablation: no hold)' if k.endswith('|simple') else ''}, {g['label']}: "
                     f"cut-off {fpct(g['test']['cutoff_rate'])} vs {fpct(g['baseline_test']['cutoff_rate'])}, "
                     f"p50 {a50} vs {b50} ms, p95 {a95} vs {b95} ms "
                     f"-> **{'met' if g['met'] else 'not met'}**")
        else:
            L.append(f"* {nice(k.split('|')[0])}: **not met** ({g['reason']})")
    L.append("")
    for var, title in VARIANTS.items():
        h = heads[(heads.variant == var) & (~heads.oracle)]
        if h.empty:
            continue
        L += [f"## {title} (n = {int(h.n.dropna().iloc[0])} turns)", ""] + headline_table(h, compact=False)
        L += ["", "Paired differences against the matched Silero baseline (same test turns, same resamples):", ""]
        L += delta_table(dl, var, compact=False) + [""]
    L += ["## Oracle rows (settings picked on the test set; NOT a valid comparison)", "",
          "Best prosody + streaming-ASR setting chosen on the test set itself under the Silero 500 ms test",
          "latencies. Shown only to bound what better operating-point selection could reach. Because it",
          "must meet the test latencies exactly, it can be worse than a train-picked setting that misses them",
          "by a hair.", "",
          "| test set | setting | cut-off | p50 | p95 |", "|---|---|---:|---:|---:|"]
    for _, r in heads[heads.oracle].iterrows():
        L.append(f"| {VARIANTS[r.variant]} | {r.config} | {cell(r, 'cutoff_rate', True)} | "
                 f"{cell(r, 'latency_p50_ms', False)} | {cell(r, 'latency_p95_ms', False)} |")
    L += ["", "## Compute (measured on this machine)", ""] + compute_table(timing) + [""]
    key_systems = ["silence_silero", "smart_turn_v3", "hazard:rt_fusion_asr@combined", "hazard:rt_fusion_asr@edacc_only"]
    for dim in ["max_pause_ms", "inserted_pause_ms", "filler", "split_kind", "accent_group"]:
        s = slices[(slices.slice == dim) & slices.system.isin(key_systems)]
        if s.empty:
            continue
        L += [f"## Cut-off rate by {dim} (cells: cut-off / median latency ms)", ""]
        for var, title in VARIANTS.items():
            sv = s[s.variant == var]
            if sv.empty:
                continue
            vals = sorted(sv.value.unique())
            counts = {v: int(sv[sv.value == v].n.iloc[0]) for v in vals}
            L += [f"**{title}**", "", "| system | setting | " + " | ".join(f"{v} (n={counts[v]})" for v in vals) + " |",
                  "|---|---|" + "---:|" * len(vals)]
            for (system, config), g in sv.groupby(["system", "config"], sort=False):
                g = g.set_index("value")
                L.append(f"| {nice(system)} | {config.split(' [')[0]} | " + " | ".join(
                    fpct(g.cutoff_rate.get(v, np.nan)) + " / " + fms(g.latency_p50_ms.get(v, np.nan)) for v in vals) + " |")
            L.append("")
    L += ["## Operating points chosen on training data", "", "```",
          json.dumps({d: {k: {T: (x or {}).get("param") for T, x in v.items()} for k, v in o["chosen"].items()}
                      for d, o in ops.items()}, indent=1), "```", ""]
    open("results/RESULTS.md", "w").write("\n".join(L))

    if os.path.exists("README.md"):
        text = open("README.md").read()
        for var in VARIANTS:
            h = heads[(heads.variant == var) & (~heads.oracle)]
            text = replace_block(text, var, headline_table(h, compact=True))
            text = replace_block(text, f"deltas_{var}", delta_table(dl, var, compact=True))
        text = replace_block(text, "compute", compute_table(timing))
        gl = []
        for k, g in goal.items():
            if "test" in g:
                dc = g["delta_ci"]["cutoff_rate"]
                a50, b50 = fms_cmp(g["test"]["latency_p50_ms"], g["baseline_test"]["latency_p50_ms"])
                a95, b95 = fms_cmp(g["test"]["latency_p95_ms"], g["baseline_test"]["latency_p95_ms"])
                gl.append(f"* {nice(k.split('|')[0])}{' (ablation: no hold)' if k.endswith('|simple') else ''} "
                          f"({g['label']}): cut-off {fpct(g['test']['cutoff_rate'])} vs "
                          f"{fpct(g['baseline_test']['cutoff_rate'])}, p50 {a50} vs {b50} ms, "
                          f"p95 {a95} vs {b95} ms; paired Δ cut-off "
                          f"{100 * dc[0]:+.1f} pp (95% CI {100 * dc[1]:+.1f} to {100 * dc[2]:+.1f}): "
                          f"**{'met' if g['met'] else 'not met'}**")
            else:
                gl.append(f"* {nice(k.split('|')[0])}: **not met** ({g['reason']})")
        text = replace_block(text, "goal", gl)
        open("README.md", "w").write(text)
    print("wrote results/RESULTS.md, results/plots/ and README tables")


if __name__ == "__main__":
    main()
