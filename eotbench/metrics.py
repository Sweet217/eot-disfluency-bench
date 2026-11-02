"""End-of-turn metrics.

Each system emits a single *first fire* time per item (the moment it would hand
the floor to the assistant). Against the ground-truth turn end ``t_end``:

* premature cut-off: fire < t_end (the system would have interrupted the user);
* latency: fire - t_end for items that were not cut off;
* no fire inside the observation window counts as a miss (latency = +inf);
* very late: latency > LATE_MS (or miss).

Latency percentiles are computed over the items that were *not* cut off, with
misses included as +inf, so p95 becomes +inf if more than 5% of them are missed.
"""
from __future__ import annotations

import math

import numpy as np

LATE_MS = 2000.0


def item_outcome(fire: float | None, t_end: float) -> dict:
    if fire is None or (isinstance(fire, float) and math.isnan(fire)):
        return dict(premature=False, latency_ms=math.inf, miss=True, late=True)
    if fire < t_end:
        return dict(premature=True, latency_ms=math.nan, miss=False, late=False)
    lat = (fire - t_end) * 1000.0
    return dict(premature=False, latency_ms=lat, miss=False, late=lat > LATE_MS)


def _pct(v: np.ndarray, q: float) -> float:
    if len(v) == 0:
        return math.nan
    # 'higher' keeps +inf when the requested rank falls on a miss
    return float(np.percentile(v, q, method="higher"))


def summarize(fires: list, t_ends: list) -> dict:
    outs = [item_outcome(f, t) for f, t in zip(fires, t_ends)]
    n = len(outs)
    prem = np.array([o["premature"] for o in outs])
    lat = np.array([o["latency_ms"] for o in outs if not o["premature"]], float)
    return dict(
        n=n,
        cutoff_rate=float(prem.mean()) if n else math.nan,
        latency_p50_ms=_pct(lat, 50),
        latency_p95_ms=_pct(lat, 95),
        late_or_miss_rate=float(np.mean([o["late"] for o in outs])) if n else math.nan,
        miss_rate=float(np.mean([o["miss"] for o in outs])) if n else math.nan,
    )
