"""Event-driven simulation of a streaming ASR running next to the endpointer.

One decoder worker serves the stream. Audio arrives in real time, so audio time
equals wall-clock time. The worker re-transcribes a rolling window:

* while the user is speaking, every `hop_s` of new audio (if the worker is idle);
* as soon as a silence starts (first silent VAD frame), so that the transcript
  that ends at the pause is ready early in the pause; if the worker is busy, it
  runs as soon as it is free, provided the silence is still going on.

Each decode's *measured* wall-clock cost (plus the cost of scoring the new text)
decides when its transcript becomes available. The endpointer may only use a
transcript at frames after its availability time; nothing is assumed instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Transcript:
    launch_t: float            # audio/wall time the decode started (window ends here)
    avail_t: float             # time the text and its features are available
    text: str
    feats: dict = field(default_factory=dict)
    decode_s: float = 0.0
    score_s: float = 0.0
    kind: str = "rolling"      # "rolling" | "onset"


def schedule(speech: list[bool] | "np.ndarray", frame_s: float, armed_from: int | None,
             decode: Callable[[float], tuple[str, float]],
             score: Callable[[str], tuple[dict, float]],
             hop_s: float = 0.5) -> list[Transcript]:
    """Run the decoder schedule over a VAD frame sequence.

    speech: per-frame speech state (True = speech) as the endpointer sees it.
    armed_from: first frame index at which the endpointer is armed (None = never).
    decode(t_end) -> (text, seconds): transcribe the window ending at t_end; returns measured cost.
    score(text) -> (features, seconds): text features and their measured cost.
    The state of frame i is known at time (i + 1) * frame_s.
    """
    out: list[Transcript] = []
    if armed_from is None:
        return out
    free_at = 0.0
    last_launch = -1e9
    pending_onset = False
    last_speech_t = None
    for i in range(armed_from, len(speech)):
        t = (i + 1) * frame_s
        if speech[i]:
            last_speech_t = t
            pending_onset = False
        elif i > 0 and speech[i - 1]:
            pending_onset = True      # a silence just started
        if free_at > t + 1e-9:
            continue
        kind = None
        if pending_onset and not speech[i]:
            kind = "onset"
        elif speech[i] and t - last_launch >= hop_s - 1e-9:
            kind = "rolling"
        if kind is None:
            continue
        text, c_dec = decode(t)
        feats, c_sc = score(text)
        out.append(Transcript(launch_t=t, avail_t=t + c_dec + c_sc, text=text, feats=feats,
                              decode_s=c_dec, score_s=c_sc, kind=kind))
        free_at = t + c_dec + c_sc
        last_launch = t
        if kind == "onset":
            pending_onset = False
    return out


def latest_available(transcripts: list[Transcript], t: float) -> Transcript | None:
    best = None
    for tr in transcripts:
        if tr.avail_t <= t + 1e-9 and (best is None or tr.avail_t >= best.avail_t):
            best = tr
    return best
