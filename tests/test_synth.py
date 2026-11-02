import shutil

import numpy as np
import pytest

from eotbench import audio, synth

pytestmark = pytest.mark.skipif(shutil.which("say") is None, reason="macOS `say` not available")


def _spec(**kw):
    base = dict(id="t0", split="test", voice="Daniel", accent_group="en_native", sentence_idx=0,
                split_kind="mid_clause", has_pause=True, pause_ms=800, filler="um", snr_db=40, seed=1)
    base.update(kw)
    return synth.SynthSpec(**base)


@pytest.mark.parametrize("pause_ms", [300, 1200])
def test_inserted_pause_and_turn_end_ground_truth(tmp_path, pause_ms):
    (tmp_path / "wav16").mkdir()
    (tmp_path / "wav8").mkdir()
    rec = synth.render(_spec(pause_ms=pause_ms), str(tmp_path), telephony=True)
    a_end, b_on = rec["inserted_pause"]
    # the inserted pause has exactly the requested length (10 ms trimming resolution)
    assert abs((b_on - a_end) * 1000 - pause_ms) <= 1
    # the labelled pause list contains it
    assert any(abs(s - a_end) <= 0.02 and abs(e - b_on) <= 0.02 for s, e in rec["pauses"])
    # turn end: speech energy before t_end, only noise after it
    x, sr = audio.read(rec["path"])
    db = audio.frame_db(x, sr)
    t = int(rec["t_end"] / audio.FRAME_S)
    assert db[t - 10:t].max() > db[t + 5:t + 300].max() + 15
    assert np.isclose(rec["duration"], rec["t_end"] + synth.TAIL_S, atol=0.02)
    # inserted pause is silent apart from the added noise floor
    i0, i1 = int(a_end / audio.FRAME_S) + 2, int(b_on / audio.FRAME_S) - 2
    assert db[i0:i1].max() < db[t - 10:t].max() - 15
    # telephony copy is 8 kHz with the same duration
    x8, sr8 = audio.read(rec["path8"])
    assert sr8 == 8000 and abs(len(x8) / sr8 - len(x) / sr) < 0.01
    # oracle words: the prefix ends at the pause, the rest after it
    pre = [w for w, e in rec["words"] if e <= a_end + 1e-3]
    assert pre[-1] == "um"
    assert rec["words"][-1][1] == pytest.approx(rec["t_end"], abs=1e-3)


def test_control_item_has_no_inserted_pause(tmp_path):
    (tmp_path / "wav16").mkdir()
    (tmp_path / "wav8").mkdir()
    rec = synth.render(_spec(has_pause=False, pause_ms=0, filler=None), str(tmp_path), telephony=False)
    assert rec["inserted_pause"] is None
    assert "," not in rec["text"]


def test_splits_are_disjoint():
    vs = synth.voice_split()
    tr = {v for v, s in vs.items() if s == "train"}
    te = {v for v, s in vs.items() if s == "test"}
    assert tr and te and not (tr & te)
    for g, voices in synth.VOICES.items():
        assert any(vs[v] == "test" for v in voices), g
    specs = synth.make_specs(50, 50)
    assert not ({s.voice for s in specs if s.split == "train"} & {s.voice for s in specs if s.split == "test"})
    assert not ({s.sentence_idx for s in specs if s.split == "train"} &
                {s.sentence_idx for s in specs if s.split == "test"})
