"""
Do the filters help? Logistic-regression ablation with repeated splits and
paired comparisons.

Folds: each human flow cell held out once (3 folds) x R random species-stratified
partitions of the pathogen reads (different seeds) = 3R evaluations per config.
Configs are compared on identical test reads, so differences are paired.

For each comparison A vs B we report the mean AUC difference over evaluations,
how many evaluations favour A, and a paired bootstrap 95% CI of the AUC
difference (pooled across the 3 flow-cell folds of partition 0).

Configs: stats (70), filters (210), full (280), stats+one filter family,
full minus one family, and full with all 15 filters' pooled-max features only.
Checkpointed; exit code 3 = rerun.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from real_data_loader import SPECIES
from ablation import cols, FILTER_GROUPS, N_FILT

GROUPS = list(FILTER_GROUPS)


def feature_sets():
    stats = np.arange(N_FILT, N_FILT + 70)
    allf = np.arange(N_FILT + 70)
    fs = {"stats": stats, "filters": np.arange(N_FILT), "full": allf}
    for g in GROUPS:
        fs[f"stats+{g}"] = np.concatenate([cols([g]), stats])
        fs[f"drop_{g}"] = np.setdiff1d(allf, cols([g]))
    return fs


COMPARISONS = [("full", "stats"), ("full", "filters"), ("full", "drop_edges"),
               ("full", "drop_log"), ("full", "drop_sinusoids"), ("full", "drop_moving_avg"),
               ("stats+edges", "stats"), ("stats+log", "stats"),
               ("stats+sinusoids", "stats"), ("stats+moving_avg", "stats")]


def fit_predict(A_tr, y_tr, A_te, C):
    sc = StandardScaler().fit(A_tr)
    m = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(A_tr), y_tr)
    return m.predict_proba(sc.transform(A_te))[:, 1]


def paired_boot(y, pa, pb, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/fs_cache"))
    ap.add_argument("--out", default="results/revision/ablation_lr.json")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=140)
    ap.add_argument("--seed", type=int, default=42)
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
    fcs = sorted(set(fc[y == 0]))
    fsets = feature_sets()

    out = json.load(open(a.out)) if os.path.exists(a.out) else {"auc": {}, "acc": {}, "boot": {}}
    probs0 = {}   # partition-0 probabilities for the paired bootstrap
    pending = False
    for rep in range(a.reps):
        rng = np.random.RandomState(a.seed + 100 + rep)
        part = np.full(len(y), -1)
        for s in SPECIES:
            idx = rng.permutation(np.where(src == s)[0])
            for k, ch in enumerate(np.array_split(idx, 3)):
                part[ch] = k
        for k, f in enumerate(fcs):
            key = f"{rep}|{f}"
            if key in out["auc"] and (rep > 0 or key in out["boot"]):
                continue
            if time.time() - t0 > a.budget:
                pending = True
                continue
            te = ((y == 0) & (fc == f)) | ((y == 1) & (part == k))
            res, acc, pr = {}, {}, {}
            for name, c in fsets.items():
                p = fit_predict(F[~te][:, c], y[~te], F[te][:, c], a.C)
                res[name] = float(roc_auc_score(y[te], p))
                acc[name] = float(accuracy_score(y[te], p >= 0.5))
                pr[name] = p
            out["auc"][key], out["acc"][key] = res, acc
            if rep == 0:
                out["boot"][key] = {f"{x}-{z}": paired_boot(y[te], pr[x], pr[z])
                                    for x, z in COMPARISONS}
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(out, open(a.out, "w"))
            print(key, f"full {res['full']:.4f} stats {res['stats']:.4f}", flush=True)
    sys.exit(3 if pending else 0)


if __name__ == "__main__":
    main()
