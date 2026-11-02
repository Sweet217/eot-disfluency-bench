"""Build the benchmark: synthetic TTS items + EdAcc real conversational items.

Usage: python scripts/prepare_data.py [--n-train 1000 --n-test 480 --edacc-row-groups 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eotbench import audio, edacc, synth  # noqa: E402
from eotbench.features import asr_words  # noqa: E402

DATA = "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=480)
    ap.add_argument("--edacc-row-groups", type=int, nargs="+", default=[2])
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-edacc", action="store_true")
    args = ap.parse_args()

    if not args.skip_synth:
        t = time.time()
        recs = synth.build(os.path.join(DATA, "synth"), args.n_train, args.n_test)
        print(f"[synth] {len(recs)} items in {time.time() - t:.0f}s")

    if not args.skip_edacc:
        t = time.time()
        raw = edacc.fetch_row_groups(os.path.join(DATA, "raw", "edacc_rg"), args.edacc_row_groups)
        recs = edacc.build(raw, os.path.join(DATA, "edacc"))
        print(f"[edacc] {len(recs)} turn items from {len(raw)} row groups in {time.time() - t:.0f}s")
        # gold-transcript word timings via alignment to hindsight ASR word timestamps
        t = time.time()
        for r in recs:
            x, sr = audio.read(r["path"])
            hyp = asr_words(x)
            r["words"] = edacc.align_words(r["text"].split(), hyp, r["t_start"], r["t_end"])
        print(f"[edacc] aligned words in {time.time() - t:.0f}s")
        with open(os.path.join(DATA, "edacc", "manifest.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
