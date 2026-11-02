import math

import numpy as np

from eotbench import metrics, systems, vad
from eotbench.edacc import align_words, is_backchannel


def test_item_outcome_premature_and_latency():
    assert metrics.item_outcome(1.0, 2.0)["premature"] is True
    o = metrics.item_outcome(2.5, 2.0)
    assert not o["premature"] and math.isclose(o["latency_ms"], 500.0)
    assert not o["late"]
    assert metrics.item_outcome(4.5, 2.0)["late"]  # 2.5 s > LATE_MS
    o = metrics.item_outcome(None, 2.0)
    assert o["miss"] and o["late"] and math.isinf(o["latency_ms"])


def test_fire_exactly_at_end_is_not_premature():
    o = metrics.item_outcome(2.0, 2.0)
    assert not o["premature"] and o["latency_ms"] == 0.0


def test_summarize_rates_and_percentiles():
    t_end = [1.0] * 10
    fires = [0.5, 0.5] + [1.1] * 4 + [1.3] * 3 + [None]
    s = metrics.summarize(fires, t_end)
    assert s["n"] == 10
    assert math.isclose(s["cutoff_rate"], 0.2)
    assert math.isclose(s["latency_p50_ms"], 300.0, abs_tol=1e-6) or math.isclose(s["latency_p50_ms"], 100.0, abs_tol=1e-6)
    assert math.isinf(s["latency_p95_ms"])  # the miss lands in the top 5 %
    assert math.isclose(s["miss_rate"], 0.1)


def test_silence_runs_and_threshold_endpointer():
    # 10 speech frames, 5 silence (pause), 10 speech, then 50 silence
    speech = np.array([1] * 10 + [0] * 5 + [1] * 10 + [0] * 50, bool)
    fs = 0.02
    runs = vad.silence_runs(speech, fs, min_speech_ms=200)
    assert runs == [(10, 5), (25, 50)]
    # 100 ms threshold fires inside the pause (premature)
    assert math.isclose(systems.silence_fire(runs, fs, 100), (10 + 5) * fs)
    # 200 ms threshold skips the 100 ms pause and fires 200 ms after the end
    assert math.isclose(systems.silence_fire(runs, fs, 200), (25 + 10) * fs)
    # threshold longer than any run -> no fire
    assert systems.silence_fire(runs, fs, 5000) is None


def test_leading_silence_and_short_blips_do_not_arm():
    speech = np.array([0] * 20 + [1] * 5 + [0] * 10, bool)
    assert vad.silence_runs(speech, 0.02, min_speech_ms=100) == [(25, 10)]
    # a 60 ms blip in leading noise does not arm a 250 ms endpointer
    speech = np.array([1] * 3 + [0] * 20 + [1] * 15 + [0] * 30, bool)
    assert vad.silence_runs(speech, 0.02, min_speech_ms=250) == [(38, 30)]


def test_hazard_fire_uses_first_crossing_and_fallback():
    c1 = dict(start=10, len=8, p_x=np.array([0.1, 0.2, 0.3]))   # mid-turn pause: stays low
    c2 = dict(start=40, len=200, p_x=np.array([0.1, 0.6, 0.9]))
    fs = vad.SILERO_FRAME_S
    f = systems.hazard_fire([c1, c2], "p_x", 0.5)
    assert math.isclose(f, (40 + systems.FIRST_FIRE + 1) * fs)
    # nothing crosses -> fallback at MAX_FRAMES of the long run
    f = systems.hazard_fire([c1, c2], "p_x", 0.99)
    assert math.isclose(f, (40 + systems.MAX_FRAMES) * fs)


def test_smart_turn_fire_rule():
    fs = vad.SILERO_FRAME_S
    cands = [dict(start=10, len=20, p_smart=0.2), dict(start=50, len=200, p_smart=0.9)]
    stop = math.ceil(systems.SMART_TURN_STOP_MS / 1000 / fs)
    assert math.isclose(systems.smart_turn_fire(cands, fs, 0.5, 3000), (50 + stop) * fs)
    # with a short fallback the first pause already triggers it
    assert math.isclose(systems.smart_turn_fire(cands, fs, 0.95, 500), (10 + math.ceil(0.5 / fs)) * fs)


def test_backchannel_and_alignment():
    assert is_backchannel("MM HMM")
    assert is_backchannel("YEAH <LAUGH>")
    assert not is_backchannel("I THINK HE WAS JUST INTERESTING")
    gold = ["i", "think", "so", "yes"]
    hyp = [("I", 1.0), ("think", 1.4), ("yes.", 2.5)]
    w = align_words(gold, hyp, 0.5, 3.0)
    assert [x[0] for x in w] == gold
    assert w[0][1] == 1.0 and w[1][1] == 1.4 and w[3][1] == 2.5
    assert 1.4 <= w[2][1] <= 2.5  # interpolated
