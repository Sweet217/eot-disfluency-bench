import math

import numpy as np

from eotbench import edacc, features, hazard, metrics, streaming, systems, vad

FS = vad.SILERO_FRAME_S


def _fake(cost_dec=0.05, cost_score=0.01):
    calls = []

    def decode(t):
        calls.append(t)
        return f"text@{t:.3f}", cost_dec

    def score(text):
        return {"len": len(text)}, cost_score

    return decode, score, calls


def test_schedule_rolling_and_onset_decodes():
    # 40 speech frames (1.28 s) then 30 silence frames
    speech = np.array([True] * 40 + [False] * 30)
    decode, score, calls = _fake()
    trs = streaming.schedule(speech, FS, armed_from=8, decode=decode, score=score, hop_s=0.5)
    rolling = [t for t in trs if t.kind == "rolling"]
    onset = [t for t in trs if t.kind == "onset"]
    # rolling decodes at most every hop while speaking, none before arming
    launches = [t.launch_t for t in rolling]
    assert launches[0] == 9 * FS
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(launches, launches[1:]))
    # exactly one onset decode, launched at the first silent frame (its state is known at (40+1)*FS)
    assert len(onset) == 1 and math.isclose(onset[0].launch_t, 41 * FS)
    # availability = launch + decode + score cost
    assert math.isclose(onset[0].avail_t, 41 * FS + 0.06)


def test_onset_decode_waits_for_busy_decoder():
    speech = np.array([True] * 20 + [False] * 30)
    decode, score, calls = _fake(cost_dec=0.2, cost_score=0.0)
    trs = streaming.schedule(speech, FS, armed_from=8, decode=decode, score=score, hop_s=0.3)
    onset = [t for t in trs if t.kind == "onset"][0]
    prev = [t for t in trs if t.kind == "rolling"][-1]
    # the decoder was busy at the silence onset, so the onset decode starts when it is free
    assert onset.launch_t >= prev.avail_t - 1e-9
    assert onset.launch_t > 21 * FS


def test_no_decodes_when_never_armed():
    decode, score, calls = _fake()
    assert streaming.schedule(np.array([False] * 10), FS, None, decode, score) == []
    assert calls == []


def test_latest_available():
    trs = [streaming.Transcript(0.1, 0.2, "a"), streaming.Transcript(0.5, 0.7, "b")]
    assert streaming.latest_available(trs, 0.15) is None
    assert streaming.latest_available(trs, 0.3).text == "a"
    assert streaming.latest_available(trs, 0.7).text == "b"


def _cand():
    zeros_p = dict.fromkeys(features.FAST_PROSODY_FEATURES, 1.0)
    zeros_t = dict.fromkeys(features.TIMING_FEATURES, 2.0)
    tf = dict.fromkeys(features.TEXT_FEATURES, 0.0)
    return dict(start=100, len=40, is_end=True, onset=100 * FS,
                f_prosody_rt=zeros_p, f_timing=zeros_t, prosody_avail=(100 + 8) * FS,
                text_versions=[dict(avail=(100 + 2) * FS, launch=95 * FS, feats=dict(tf, n_words=3.0)),
                               dict(avail=(100 + 10) * FS, launch=101 * FS, feats=dict(tf, n_words=7.0))],
                f_text_oracle=tf, oracle_avail=(100 + 5) * FS)


def test_realtime_features_respect_availability():
    c = _cand()
    X, ks = hazard.expand(c, "rt_fusion_asr")
    cols = hazard.columns("rt_fusion_asr")
    assert ks[0] == systems.FIRST_FIRE
    p0 = cols.index("p_f0_slope_st_s")
    nw = cols.index("t_n_words")
    age = cols.index("t_age_s")
    t = (100 + ks) * FS
    # prosody unavailable before frame 8, available from then on
    assert np.all(np.isnan(X[ks < 8, p0])) and np.all(X[ks >= 8, p0] == 1.0)
    # stale transcript (3 words) until the fresh one (7 words) arrives at frame 10
    assert np.all(X[ks < 10, nw] == 3.0) and np.all(X[ks >= 10, nw] == 7.0)
    assert np.allclose(X[ks >= 10, age], t[ks >= 10] - 101 * FS)
    assert X.shape[1] == len(cols)


def test_hazard_cap_fires_before_late_model_crossing():
    p = np.zeros(60)
    p[40] = 0.99  # model crossing at frame FIRST_FIRE + 40
    cands = [dict(start=10, len=200, **{"p:x": p})]
    t, why = systems.hazard_decision(cands, "p:x", 0.5, fallback_frames=systems.cap_frames(800))
    assert why == "cap" and math.isclose(t, (10 + systems.cap_frames(800)) * FS)
    t, why = systems.hazard_decision(cands, "p:x", 0.5, fallback_frames=systems.cap_frames(3000))
    assert why == "model" and math.isclose(t, (10 + systems.FIRST_FIRE + 40) * FS)


def test_timing_features():
    runs = [(10, 8), (40, 3), (60, 20), (100, 50)]
    f = features.timing_features(runs, 100, FS)
    assert f["n_prev_pauses"] == 2  # the 3-frame gap is too short to count
    assert math.isclose(f["max_prev_pause_s"], 20 * FS)
    assert math.isclose(f["since_prev_pause_s"], (100 - 80) * FS)
    assert features.timing_features(runs, 10, FS)["since_prev_pause_s"] == -1.0


def test_cluster_bootstrap_and_paired_delta():
    t_end = [1.0] * 60
    fires = [0.5 if i % 3 == 0 else 1.3 for i in range(60)]
    clusters = [i // 6 for i in range(60)]
    ci = metrics.cluster_bootstrap(fires, t_end, clusters, B=300, seed=1)
    lo, hi = ci["cutoff_rate"]
    assert lo <= 1 / 3 <= hi
    d = metrics.paired_delta_bootstrap(fires, fires, t_end, clusters, B=100)
    assert d["cutoff_rate"] == (0.0, 0.0, 0.0)
    better = [1.3] * 60
    d = metrics.paired_delta_bootstrap(better, fires, t_end, clusters, B=200)
    assert math.isclose(d["cutoff_rate"][0], -1 / 3) and d["cutoff_rate"][2] < 0


def test_edacc_eligibility_ranking_and_split():
    meta = [
        {"speaker": "EDACC-C01-A", "text": "I THINK WE SHOULD GO TOMORROW"},
        {"speaker": "EDACC-C01-B", "text": "MM HMM"},                       # backchannel: previous is ambiguous
        {"speaker": "EDACC-C01-A", "text": "BECAUSE THE WEATHER IS BETTER THEN"},  # floor changes next: turn end
        {"speaker": "EDACC-C01-B", "text": "YES THAT MAKES A LOT OF SENSE"},  # next segment is another conversation
        {"speaker": "EDACC-C02-A", "text": "SO WHAT DID YOU DO TODAY"},      # floor changes next: turn end
        {"speaker": "EDACC-C02-B", "text": "NOTHING MUCH REALLY TO BE HONEST"},  # followed by a backchannel
        {"speaker": "EDACC-C02-A", "text": "OKAY"},
    ]
    idx = edacc.eligible_turns(meta)
    assert idx == [2, 4]
    rank = edacc.ranked_by_conversation(meta, idx, seed=3)
    assert rank[2] == 0 and rank[4] == 0  # one eligible turn per conversation
    assert rank == edacc.ranked_by_conversation(meta, idx, seed=3)  # deterministic
    recs = [dict(conversation=f"C{i % 8}", accent_group="g1" if i % 2 else "g2") for i in range(80)]
    edacc.assign_split(recs)
    by = {}
    for r in recs:
        by.setdefault(r["conversation"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by.values())  # conversation-disjoint
    assert {r["split"] for r in recs} == {"train", "test"}


def test_confident_hold_extends_wait_only_while_model_is_confident():
    fs = FS
    n500, n2000 = systems.cap_frames(500), systems.cap_frames(2000)
    k_rise = 40  # P rises above the hold threshold at frame 40 (~1.3 s)
    p = np.full(100 - systems.FIRST_FIRE + 1, 0.01)
    p[k_rise - systems.FIRST_FIRE:] = 0.5
    cands = [dict(start=0, len=100, **{"p:x": p})]
    # plain max wait: fires at 500 ms
    t, why = systems.hazard_decision(cands, "p:x", 0.9, fallback_frames=n500)
    assert why == "cap" and math.isclose(t, n500 * fs)
    # with a hold: keeps waiting while P <= 0.05, fires when confidence drops (frame 40)
    t, why = systems.hazard_decision(cands, "p:x", 0.9, fallback_frames=n500, hold_frames=n2000, hold_theta=0.05)
    assert why == "cap" and math.isclose(t, k_rise * fs)
    # hold never exceeds the hard limit
    p2 = np.full_like(p, 0.01)
    t, why = systems.hazard_decision([dict(start=0, len=100, **{"p:x": p2})], "p:x", 0.9,
                                     fallback_frames=n500, hold_frames=n2000, hold_theta=0.05)
    assert math.isclose(t, n2000 * fs)
    # a pause that ends (speech resumes) before the hold limit is never cut off
    short = dict(start=0, len=30, **{"p:x": p2[:30 - systems.FIRST_FIRE + 1]})
    assert systems.hazard_decision([short], "p:x", 0.9, fallback_frames=n500, hold_frames=n2000,
                                   hold_theta=0.05) == (None, None)
