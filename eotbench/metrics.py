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


# ------------------------------------------------------------- cluster bootstrap
STATS = ("cutoff_rate", "latency_p50_ms", "latency_p95_ms", "late_or_miss_rate")


def _outcome_arrays(fires: list, t_ends: list):
    outs = [item_outcome(f, t) for f, t in zip(fires, t_ends)]
    prem = np.array([o["premature"] for o in outs])
    lat = np.array([o["latency_ms"] for o in outs], float)
    late = np.array([o["late"] for o in outs])
    return prem, lat, late


def _stats_from(prem, lat, late, idx) -> np.ndarray:
    p = prem[idx]
    lv = lat[idx][~p]
    return np.array([p.mean(), _pct(lv, 50), _pct(lv, 95), late[idx].mean()])


def _cluster_samples(clusters: list, B: int, seed: int):
    clusters = np.asarray(clusters, dtype=object)
    uniq = sorted(set(clusters.tolist()))
    members = [np.nonzero(clusters == u)[0] for u in uniq]
    rng = np.random.default_rng(seed)
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def cluster_bootstrap(fires: list, t_ends: list, clusters: list, B: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Percentile CIs for the headline metrics, resampling whole clusters (conversations/voices)
    with replacement, so that turns from the same speakers are not treated as independent."""
    prem, lat, late = _outcome_arrays(fires, t_ends)
    sims = np.array([_stats_from(prem, lat, late, idx) for idx in _cluster_samples(clusters, B, seed)])
    lo = np.percentile(sims, 100 * alpha / 2, axis=0, method="lower")
    hi = np.percentile(sims, 100 * (1 - alpha / 2), axis=0, method="higher")
    return {s: (float(lo[k]), float(hi[k])) for k, s in enumerate(STATS)}


def paired_delta_bootstrap(fires_a: list, fires_b: list, t_ends: list, clusters: list, B: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and CI of (metric_a - metric_b) on the same items, same cluster resamples."""
    a = _outcome_arrays(fires_a, t_ends)
    b = _outcome_arrays(fires_b, t_ends)
    allidx = np.arange(len(t_ends))
    with np.errstate(invalid="ignore"):
        point = _stats_from(*a, allidx) - _stats_from(*b, allidx)
        sims = np.array([_stats_from(*a, idx) - _stats_from(*b, idx)
                         for idx in _cluster_samples(clusters, B, seed)])
    lo = np.nanpercentile(sims, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(sims, 100 * (1 - alpha / 2), axis=0)
    return {s: (float(point[k]), float(lo[k]), float(hi[k])) for k, s in enumerate(STATS)}
