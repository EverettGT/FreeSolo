"""
Feature/model ablation on the main 12k human + 12k pathogen sample.

Question: is the fixed filter bank + MLP better than simpler alternatives?
Every model sees the same reads and the same flow-cell-held-out folds as
diagnostics.py stage H1 (each human flow cell held out once, with a
species-stratified third of the pathogen reads).

Models
  raw        MLP on the normalised 4,000-sample window (no features)
  stats      MLP on the 70 statistical features (columns 210-279)
  filters    MLP on the 210 filter features (columns 0-209)
  full       MLP on all 280 features (the FreeSolo classifier)
  logreg     logistic regression on all 280 features
  logreg_raw logistic regression on the normalised 4,000-sample window
  drop_*     full model with one filter family removed

Checkpointed: rerun until it exits 0 (exit code 3 = more work remains).
Requires the caches written by diagnostics.py (main_*.npz/json, main_feat_*.npy).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from diagnostics import summarize
from real_data_loader import SPECIES

FILTER_GROUPS = {"edges": range(0, 4), "log": range(4, 7),
                 "sinusoids": range(7, 12), "moving_avg": range(12, 15)}
PER_FILTER = 14
N_FILT = 15 * PER_FILTER


def cols(groups):
    out = []
    for g in groups:
        for f in FILTER_GROUPS[g]:
            out += list(range(f * PER_FILTER, (f + 1) * PER_FILTER))
    return np.array(out)


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                         alpha=1e-4, batch_size=64, learning_rate="adaptive",
                         learning_rate_init=1e-3, max_iter=200, random_state=seed,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)


def fit_predict(kind, A_tr, y_tr, A_te, seed):
    sc = StandardScaler().fit(A_tr)
    m = LogisticRegression(max_iter=2000, C=1.0) if kind == "logreg" else mlp(seed)
    m.fit(sc.transform(A_tr), y_tr)
    return m.predict_proba(sc.transform(A_te))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/fs_cache"))
    ap.add_argument("--out", default="results/revision/ablation.json")
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="full,stats,filters,logreg,raw,"
                    "drop_edges,drop_log,drop_sinusoids,drop_moving_avg,logreg_raw")
    a = ap.parse_args()
    t0 = time.time()

    mains = ["human"] + list(SPECIES)
    meta = []
    for g in mains:
        meta += json.load(open(os.path.join(a.cache, f"main_{g}.json")))["meta"]
    y = np.array([0 if m["source"] == "human" else 1 for m in meta])
    fc = np.array([m["flow_cell"] for m in meta])
    src = np.array([m["source"] for m in meta])
    n_ch = int(np.ceil(len(y) / 3000))
    F = np.concatenate([np.load(os.path.join(a.cache, f"main_feat_{i}.npy")) for i in range(n_ch)])
    assert len(F) == len(y)

    # same folds as diagnostics.py H1 (seed + 7)
    rng = np.random.RandomState(a.seed + 7)
    part = np.full(len(y), -1)
    for s in SPECIES:
        idx = rng.permutation(np.where(src == s)[0])
        for k, ch in enumerate(np.array_split(idx, 3)):
            part[ch] = k
    fcs = sorted(set(fc[y == 0]))

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    X_raw = None
    pending = False
    for kind in a.models.split(","):
        for k, f in enumerate(fcs):
            key = f"{kind}|{f}"
            if key in res:
                continue
            if time.time() - t0 > a.budget:
                pending = True
                continue
            te = ((y == 0) & (fc == f)) | ((y == 1) & (part == k))
            if kind in ("raw", "logreg_raw"):
                if X_raw is None:
                    X_raw = np.concatenate([np.load(os.path.join(a.cache, f"main_{g}.npz"))["x"]
                                            for g in mains]).astype(np.float32)
                A, name = X_raw, "raw"
            elif kind == "stats":
                A = F[:, N_FILT:]
            elif kind == "filters":
                A = F[:, :N_FILT]
            elif kind in ("full", "logreg"):
                A = F
            elif kind.startswith("drop_"):
                A = np.delete(F, cols([kind[5:]]), axis=1)
            else:
                raise SystemExit(f"unknown model {kind}")
            s0 = time.time()
            clf = "logreg" if kind in ("logreg", "logreg_raw") else "mlp"
            p = fit_predict(clf, A[~te], y[~te], A[te], a.seed)
            r = summarize(y[te], p)
            r["n_features"] = int(A.shape[1])
            r["seconds"] = time.time() - s0
            res[key] = r
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=2)
            print(f"{key}: AUC {r['auc']:.4f} acc {r['accuracy']:.4f} ({r['seconds']:.0f}s)", flush=True)
    sys.exit(3 if pending else 0)


if __name__ == "__main__":
    main()
