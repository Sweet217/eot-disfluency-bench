"""Frame-level voice activity used by every endpointer (all processing is causal)."""
from __future__ import annotations

import functools

import numpy as np

SILERO_FRAME = 512          # samples at 16 kHz = 32 ms
SILERO_FRAME_S = SILERO_FRAME / 16000
WEBRTC_FRAME_S = 0.030


@functools.lru_cache(maxsize=1)
def _silero():
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    return load_silero_vad()


def silero_probs(x: np.ndarray) -> np.ndarray:
    """Streaming Silero VAD speech probability per 32 ms frame (16 kHz input)."""
    import torch

    m = _silero()
    m.reset_states()
    n = len(x) // SILERO_FRAME
    out = np.empty(n, np.float32)
    with torch.no_grad():
        for i in range(n):
            out[i] = m(torch.from_numpy(x[i * SILERO_FRAME:(i + 1) * SILERO_FRAME].copy()), 16000).item()
    return out


def hysteresis(p: np.ndarray, on: float = 0.5, off: float = 0.35) -> np.ndarray:
    """Silero's recommended speech state machine: enter speech at p>=on, leave at p<off."""
    s = np.zeros(len(p), bool)
    state = False
    for i, v in enumerate(p):
        if not state and v >= on:
            state = True
        elif state and v < off:
            state = False
        s[i] = state
    return s


def webrtc_speech(x: np.ndarray, aggressiveness: int = 2) -> np.ndarray:
    import webrtcvad

    v = webrtcvad.Vad(aggressiveness)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    hop = int(WEBRTC_FRAME_S * 16000)
    n = len(pcm) // hop
    return np.array([v.is_speech(pcm[i * hop:(i + 1) * hop].tobytes(), 16000) for i in range(n)], bool)


def silence_runs(speech: np.ndarray, frame_s: float, min_speech_ms: float = 250.0) -> list[tuple[int, int]]:
    """Silence runs (start_frame, length_frames) once the endpointer is armed.

    The endpointer arms after the first stretch of *continuous* speech of at
    least `min_speech_ms` (like Silero's min_speech_duration_ms), so VAD warm-up
    blips in leading noise cannot trigger it. The final run is open-ended.
    """
    need = int(np.ceil(min_speech_ms / 1000 / frame_s - 1e-9))
    runs, armed, cur, s = [], False, 0, None
    for i, v in enumerate(speech):
        if v:
            cur += 1
            if cur >= need:
                armed = True
            if s is not None:
                runs.append((s, i - s))
                s = None
        else:
            cur = 0
            if armed and s is None:
                s = i
    if s is not None:
        runs.append((s, len(speech) - s))
    return runs


def armed_index(speech: np.ndarray, frame_s: float, min_speech_ms: float = 250.0) -> int | None:
    """First frame at which the endpointer is armed (same rule as `silence_runs`)."""
    need = int(np.ceil(min_speech_ms / 1000 / frame_s - 1e-9))
    cur = 0
    for i, v in enumerate(speech):
        cur = cur + 1 if v else 0
        if cur >= need:
            return i
    return None
