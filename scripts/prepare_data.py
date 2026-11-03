"""Build the benchmark: synthetic TTS items + EdAcc real conversational items.

Usage: python scripts/prepare_data.py [--n-train 1000 --n-test 480 --edacc-cap 60]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eotbench import audio, edacc, synth  # noqa: E402
from eotbench.features import mlx_words  # noqa: E402

DATA = "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=480)
    ap.add_argument("--edacc-cap", type=int, default=edacc.CAP_PER_CONVERSATION,
                    help="max turns kept per EdAcc conversation")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-edacc", action="store_true")
    args = ap.parse_args()

    if not args.skip_synth:
        t = time.time()
        recs = synth.build(os.path.join(DATA, "synth"), args.n_train, args.n_test)
        print(f"[synth] {len(recs)} items in {time.time() - t:.0f}s")

    if not args.skip_edacc:
        t = time.time()
        shards = edacc.fetch_full(os.path.join(DATA, "raw", "edacc_full"))
        out = os.path.join(DATA, "edacc")
        shutil.rmtree(out, ignore_errors=True)
        recs = edacc.build(shards, out, cap=args.edacc_cap)
        print(f"[edacc] {len(recs)} turn items from {len({r['conversation'] for r in recs})} "
              f"conversations in {time.time() - t:.0f}s")
        # gold-transcript word timings via alignment to hindsight ASR word timestamps
        t = time.time()
        for r in recs:
            x, sr = audio.read(r["path"])
            r["words"] = edacc.align_words(r["text"].split(), mlx_words(x), r["t_start"], r["t_end"])
        print(f"[edacc] aligned words in {time.time() - t:.0f}s")
        with open(os.path.join(out, "manifest.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
