"""Real L2 / accented conversational items from EdAcc (CC BY-SA 4.0).

The Hugging Face release (edinburghcstr/edacc) stores one row per transcribed
speaker segment, in conversation order, without absolute timestamps. We use
segment order to decide which segments end a *turn*:

* a segment is a candidate if it is not itself a backchannel and has >= 4 words;
* it is kept only if the next segment is by the *other* speaker and is not a
  backchannel (i.e. the floor really changed hands). Segments followed by a
  backchannel are ambiguous (the speaker may have continued) and are dropped.

Within a kept segment, the true turn end is the last speech frame and the
intra-turn pauses are internal low-energy runs >= 150 ms, both found with an
offline (non-causal, hindsight) energy detector that is independent of every
system under test. Four seconds of the segment's own room tone are appended
after the turn end so endpointers have time to fire.

Only a few parquet *row groups* are fetched per shard (HTTP range requests),
keeping the download modest.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

from . import audio

REPO = "edinburghcstr/edacc"
LEAD_S = 0.5
TAIL_S = 4.0
PAUSE_MIN_S = 0.15
BACKCHANNEL = {"mm", "hmm", "mhm", "yeah", "yes", "okay", "ok", "uh", "um", "huh", "right", "oh",
               "ah", "yep", "no", "wow", "sure", "cool", "alright", "hm", "mmhmm", "nice", "true", "exactly"}
FILLER_TOKENS = {"um", "uh", "er", "erm", "hmm", "ah", "eh", "mm"}


def norm_words(text: str) -> list[str]:
    t = re.sub(r"<[^>]+>", " ", text)
    t = t.replace("MM HMM", "MMHMM")
    return [w for w in re.sub(r"[^a-z' ]", " ", t.lower()).split() if w]


def is_backchannel(text: str) -> bool:
    w = norm_words(text)
    return len(w) <= 3 and all(x in BACKCHANNEL for x in w)


def fetch_row_groups(raw_dir: str, row_groups: list[int]) -> list[str]:
    """Download selected row groups of every shard into small local parquet files."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    os.makedirs(raw_dir, exist_ok=True)
    out = []
    for f in sorted(fs.glob(f"datasets/{REPO}/data/*.parquet")):
        base = os.path.basename(f).split("-of-")[0]
        for rg in row_groups:
            dst = os.path.join(raw_dir, f"{base}_rg{rg}.parquet")
            if not os.path.exists(dst):
                with fs.open(f) as fh:
                    pf = pq.ParquetFile(fh)
                    if rg >= pf.metadata.num_row_groups:
                        continue
                    pq.write_table(pf.read_row_group(rg), dst)
                print(f"[edacc] fetched {dst}")
            out.append(dst)
    return out


def speech_mask(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    db = audio.frame_db(x, sr)
    lo, hi = np.percentile(db, 2), np.percentile(db, 98)
    thr = max(lo + 12.0, hi - 40.0)
    m = db > thr
    # close gaps < 60 ms, then drop speech runs < 50 ms (clicks)
    m = _fill_runs(m, value=False, max_len=6)
    m = _fill_runs(m, value=True, max_len=5)
    return m, thr


def _fill_runs(m: np.ndarray, value: bool, max_len: int) -> np.ndarray:
    m = m.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i] == value:
            j = i
            while j < n and m[j] == value:
                j += 1
            if i > 0 and j < n and j - i <= max_len:
                m[i:j] = not value
            i = j
        else:
            i += 1
    return m


def room_tone(x: np.ndarray, mask: np.ndarray, sr: int, seconds: float, rng) -> np.ndarray:
    hop = int(audio.FRAME_S * sr)
    quiet = [x[i * hop:(i + 1) * hop] for i in np.where(~mask)[0]]
    n = int(seconds * sr)
    if len(quiet) >= 20:
        quiet = np.concatenate(quiet)
        reps = int(np.ceil(n / len(quiet))) + 1
        order = [quiet if k % 2 == 0 else quiet[::-1] for k in range(reps)]  # mirror to avoid clicks
        return np.concatenate(order)[:n].astype(np.float32)
    lvl = 10 ** (np.percentile(audio.frame_db(x, sr), 5) / 20)
    return audio.pink_noise(n, rng) * lvl


def label_segment(x: np.ndarray, sr: int) -> dict | None:
    mask, thr = speech_mask(x, sr)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    t_on, t_end = idx[0] * audio.FRAME_S, (idx[-1] + 1) * audio.FRAME_S
    pauses, s = [], None
    for i in range(idx[0], idx[-1] + 1):
        if not mask[i] and s is None:
            s = i
        elif mask[i] and s is not None:
            if (i - s) * audio.FRAME_S >= PAUSE_MIN_S:
                pauses.append([round(s * audio.FRAME_S, 3), round(i * audio.FRAME_S, 3)])
            s = None
    return dict(t_start=t_on, t_end=t_end, pauses=pauses, mask=mask)


def build(raw_files: list[str], out_dir: str, min_dur=1.5, max_dur=25.0, seed=7) -> list[dict]:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "wav16"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "wav8"), exist_ok=True)
    rows = []
    for f in raw_files:
        t = pq.read_table(f)
        meta = t.drop(["audio"]).to_pylist()
        aud = t.column("audio").to_pylist()
        for m, a in zip(meta, aud):
            m["_audio"] = a
            m["_src"] = os.path.basename(f)
        rows.append(meta)
    recs = []
    for rs in rows:
        for i in range(len(rs) - 1):
            r, nxt = rs[i], rs[i + 1]
            conv = r["speaker"].rsplit("-", 1)[0]
            if nxt["speaker"].rsplit("-", 1)[0] != conv or nxt["speaker"] == r["speaker"]:
                continue
            if is_backchannel(r["text"]) or is_backchannel(nxt["text"]):
                continue
            if "<DTMF>" in r["text"] or len(norm_words(r["text"])) < 4:
                continue
            x, sr = sf.read(io.BytesIO(r["_audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            x = audio.resample(x, sr, audio.SR)
            dur = len(x) / audio.SR
            if not (min_dur <= dur <= max_dur):
                continue
            lab = label_segment(x, audio.SR)
            if lab is None or lab["t_end"] - lab["t_start"] < 1.0:
                continue
            # keep the segment up to the labelled end, then 4 s of its own room tone
            end_i = int(lab["t_end"] * audio.SR)
            tone = room_tone(x, lab["mask"], audio.SR, LEAD_S + TAIL_S, rng)
            lead = tone[: int(LEAD_S * audio.SR)]
            tail = tone[int(LEAD_S * audio.SR):]
            y = np.concatenate([lead, x[:end_i], tail]).astype(np.float32)
            iid = f"edacc_{r['_audio']['path'].replace('.wav', '')}"
            words = norm_words(r["text"])
            split = None  # assigned below, conversation-disjoint
            rec = dict(id=iid, set="edacc", split=split, speaker=r["speaker"], conversation=conv,
                       accent=r["accent"], l1=r["l1"], gender=r["gender"],
                       accent_group=accent_group(r["l1"], r["accent"]), text=" ".join(words),
                       has_filler=any(w in FILLER_TOKENS for w in words),
                       t_start=round(lab["t_start"] + LEAD_S, 3), t_end=round(lab["t_end"] + LEAD_S, 3),
                       pauses=[[round(p[0] + LEAD_S, 3), round(p[1] + LEAD_S, 3)] for p in lab["pauses"]],
                       duration=round(len(y) / audio.SR, 3))
            p16 = os.path.join(out_dir, "wav16", iid + ".wav")
            audio.write(p16, y, audio.SR)
            rec["path"] = p16
            recs.append(rec)
    assign_split(recs)
    write_telephony(recs, out_dir)
    return recs


def assign_split(recs: list[dict], test_frac: float = 0.5, seed: int = 0) -> None:
    """Conversation-disjoint (hence speaker-disjoint) split, balanced for size and accent group.

    Deterministic random search over conversation assignments; minimises the deviation of the
    test share of items overall and within each accent group from `test_frac`.
    """
    convs = sorted({r["conversation"] for r in recs})
    groups = sorted({r["accent_group"] for r in recs})
    cnt = {c: np.array([sum(1 for r in recs if r["conversation"] == c and r["accent_group"] == g)
                        for g in groups], float) for c in convs}
    tot = sum(cnt.values())
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(20000):
        mask = rng.random(len(convs)) < 0.5
        te = sum((cnt[c] for c, m in zip(convs, mask) if m), np.zeros(len(groups)))
        cost = abs(te.sum() / tot.sum() - test_frac) + np.abs(te / np.maximum(tot, 1) - test_frac).sum()
        if cost < best_cost:
            best, best_cost = mask.copy(), cost
    test = {c for c, m in zip(convs, best) if m}
    for r in recs:
        r["split"] = "test" if r["conversation"] in test else "train"


def write_telephony(recs: list[dict], out_dir: str) -> None:
    for r in recs:
        if r["split"] == "test":
            p8 = os.path.join(out_dir, "wav8", r["id"] + ".wav")
            if not os.path.exists(p8):
                y, _ = audio.read(r["path"])
                audio.write(p8, audio.telephony(y), 8000)
            r["path8"] = p8
        else:
            r.pop("path8", None)


ENGLISH_L1_HINTS = ("english",)


def accent_group(l1: str, accent: str) -> str:
    """EdAcc self-/linguist-reported L1: English-L1 varieties vs English as an L2."""
    s = (l1 or "").lower()
    return "edacc_L1_english" if any(h in s for h in ENGLISH_L1_HINTS) else "edacc_L2_english"


def align_words(gold: list[str], hyp: list[tuple[str, float]], t0: float, t1: float) -> list[list]:
    """Give gold transcript words end times by aligning them to ASR word timestamps.

    Matched words take the ASR time; unmatched words are linearly interpolated.
    """
    hw = [re.sub(r"[^a-z']", "", w.lower()) for w, _ in hyp]
    sm = difflib.SequenceMatcher(a=gold, b=hw, autojunk=False)
    times = [None] * len(gold)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            times[a + k] = hyp[b + k][1]
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    anchors = [(-1, t0)] + known + [(len(gold), t1)]
    for (i0, ta), (i1, tb) in zip(anchors[:-1], anchors[1:]):
        for i in range(i0 + 1, i1):
            times[i] = ta + (tb - ta) * (i - i0) / (i1 - i0)
    out, last = [], t0
    for w, t in zip(gold, times):
        last = max(last, float(t))
        out.append([w, round(min(last, t1), 3)])
    return out
