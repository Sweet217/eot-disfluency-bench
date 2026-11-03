"""Train the hazard-model variants on the speaker-disjoint training split.

Every model is also fitted in 5 folds grouped by conversation (EdAcc) or by
sentence (synthetic), giving out-of-fold predictions on the training split that
are used only to choose operating points; the test split is never used for that.
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

CACHE = "data/cache_v2"
TRAIN = {"synth": "synth_train_16k", "edacc": "edacc_train_16k"}
MODELS = ([(f"{fs}@combined", fs, ("synth", "edacc")) for fs in hazard.FEATURE_SETS] +
          [(f"{fs}@edacc_only", fs, ("edacc",)) for fs in
           ("rt_fusion_asr", "rt_fusion_oracle", "rt_prosody", "rt_text_asr", "rt_duration_only")])


def load(names):
    items = []
    for n in names:
        with open(f"{CACHE}/{n}.pkl", "rb") as f:
            items += pickle.load(f)
    return items


def main():
    os.makedirs("data/models", exist_ok=True)
    by_domain = {d: load([n]) for d, n in TRAIN.items()}
    for name, fset, domains in MODELS:
        t = time.time()
        items = [it for d in domains for it in by_domain[d]]
        groups = np.array([hazard.group_key(it["meta"]) for it in items], dtype=object)
        model = hazard.fit(items, fset)
        oof = {}
        for tr, va in GroupKFold(n_splits=5).split(items, groups=groups):
            m = hazard.fit([items[i] for i in tr], fset)
            sub = [items[i] for i in va]
            hazard.predict_cands(m, sub, fset, "_p")
            for it in sub:
                for j, c in enumerate(it["cands"]):
                    if "_p" in c:
                        oof[(it["meta"]["id"], j)] = c.pop("_p")
        with open(f"data/models/{name}.pkl", "wb") as f:
            pickle.dump(dict(model=model, oof=oof, fset=fset, domains=domains,
                             columns=hazard.columns(fset)), f)
        print(f"[train] {name}: {len(items)} items, {len(set(groups))} groups, {time.time() - t:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
