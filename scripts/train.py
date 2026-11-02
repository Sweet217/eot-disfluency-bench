"""Train the hazard-model variants on the speaker-disjoint training split.

Also produces out-of-fold (GroupKFold by speaker) predictions on the training
split, used only to pick operating points (theta) without touching test data.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from eotbench import hazard  # noqa: E402

TRAIN = ["synth_train_16k", "edacc_train_16k"]


def load(names):
    items = []
    for n in names:
        with open(f"data/cache/{n}.pkl", "rb") as f:
            items += pickle.load(f)
    return items


def main():
    items = load(TRAIN)
    speakers = np.array([it["meta"]["speaker"] for it in items], dtype=object)
    os.makedirs("data/models", exist_ok=True)
    print(f"[train] {len(items)} training items, {len(set(speakers))} speakers/voices")
    for fset in hazard.FEATURE_SETS:
        t = time.time()
        model = hazard.fit(items, fset)
        oof = {}
        for tr, va in GroupKFold(n_splits=5).split(items, groups=speakers):
            m = hazard.fit([items[i] for i in tr], fset)
            sub = [items[i] for i in va]
            hazard.predict_cands(m, sub, fset, "_p")
            for it in sub:
                for j, c in enumerate(it["cands"]):
                    if "_p" in c:
                        oof[(it["meta"]["id"], j)] = c.pop("_p")
        with open(f"data/models/{fset}.pkl", "wb") as f:
            pickle.dump(dict(model=model, oof=oof, columns=hazard.columns(fset)), f)
        print(f"[train] {fset}: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
