"""
Do the hand-designed filters help, and does their design matter?

Uses the main 12k human + 12k pathogen sample cached by diagnostics.py and
logistic regression (the FreeSolo default classifier).

1. Random filter banks: B banks of 15 random Gaussian kernels with the same
   lengths as the designed bank (5, 10, 20, 40, 19, 43, 91, 50 x5, 5, 15, 50),
   unit-normalised, passed through the same ReLU + pooling summary
   (14 features per filter, 210 per bank). If random kernels add as much to the
   statistical features as the designed ones, the design choices do not matter.
2. Feature-set comparisons on identical test reads (paired): statistics only,
   filters only, both, statistics + one filter family, both minus one family,
   statistics + random bank b, random bank b only.

Folds: each human flow cell held out once x R species-stratified partitions of
the pathogen reads (3R evaluations per configuration). Partition 0 equals the
partition used in diagnostics.py H1. A paired bootstrap CI of each AUC
difference is computed on partition 0.

Checkpointed; exit code 3 means rerun the same command.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import Conv1DFeatureExtractor
from real_data_loader import SPECIES

PER_FILTER = 14
N_FILT = 15 * PER_FILTER
FAMILIES = {"edges": range(0, 4), "log": range(4, 7),
            "sinusoids": range(7, 12), "moving_avg": range(12, 15)}


def fam_cols(g):
    return np.concatenate([np.arange(f * PER_FILTER, (f + 1) * PER_FILTER) for f in FAMILIES[g]])


def random_bank(seed):
    fe = Conv1DFeatureExtractor()
    rng = np.random.RandomState(seed)
    bank = []
    for f in fe.filters:
        k = rng.randn(len(f))
        bank.append(k / np.linalg.norm(k))
    return bank


def fit_predict(A_tr, y_tr, A_te):
    sc = StandardScaler().fit(A_tr)
    m = LogisticRegression(max_iter=3000).fit(sc.transform(A_tr), y_tr)
    return m.predict_proba(sc.transform(A_te))[:, 1]


def paired_boot(y, pa, pb, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/fs_cache"))
    ap.add_argument("--out", default="results/revision/filter_controls.json")
    ap.add_argument("--banks", type=int, default=5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=140)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    left = lambda: a.budget - (time.time() - t0)

    mains = ["human"] + list(SPECIES)
    meta = []
    for g in mains:
        meta += json.load(open(os.path.join(a.cache, f"main_{g}.json")))["meta"]
    y = np.array([0 if m["source"] == "human" else 1 for m in meta])
    fc = np.array([m["flow_cell"] for m in meta])
    src = np.array([m["source"] for m in meta])
    F = np.concatenate([np.load(os.path.join(a.cache, f"main_feat_{i}.npy"))
                        for i in range(int(np.ceil(len(y) / 3000)))])
    stats, filt = F[:, N_FILT:], F[:, :N_FILT]

    # ---- random-bank features (chunked, cached)
    CH = 3000
    RB = {}
    X = None
    for b in range(a.banks):
        parts = []
        for i in range(int(np.ceil(len(y) / CH))):
            fn = os.path.join(a.cache, f"rand{b}_feat_{i}.npy")
            if not os.path.exists(fn):
                if left() < 40:
                    print("budget reached during random-bank features", flush=True)
                    sys.exit(3)
                if X is None:
                    X = np.concatenate([np.load(os.path.join(a.cache, f"main_{g}.npz"))["x"] for g in mains])
                fe = Conv1DFeatureExtractor()
                fe.filters = random_bank(1000 + b)
                s0 = time.time()
                np.save(fn, np.nan_to_num(np.array([fe._apply_convolutions(x)
                                                    for x in X[i * CH:(i + 1) * CH]], float)))
                print(f"random bank {b} chunk {i}: {time.time()-s0:.0f}s", flush=True)
            parts.append(np.load(fn))
        RB[b] = np.concatenate(parts)

    sets = {"stats": stats, "filters": filt, "stats+filters": F}
    for g in FAMILIES:
        sets[f"stats+{g}"] = np.hstack([filt[:, fam_cols(g)], stats])
        sets[f"stats+filters-{g}"] = np.delete(F, fam_cols(g), axis=1)
    for b in RB:
        sets[f"random{b}"] = RB[b]
        sets[f"stats+random{b}"] = np.hstack([RB[b], stats])

    comps = [("stats+filters", "stats"), ("stats+filters", "filters")]
    comps += [(f"stats+{g}", "stats") for g in FAMILIES]
    comps += [("stats+filters", f"stats+filters-{g}") for g in FAMILIES]
    comps += [(f"stats+random{b}", "stats") for b in RB]
    comps += [("stats+filters", f"stats+random{b}") for b in RB]

    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out.setdefault("auc", {}); out.setdefault("acc", {}); out.setdefault("boot", {})
    out["random_bank_kernel_lengths"] = [len(f) for f in random_bank(0)]
    fcs = sorted(set(fc[y == 0]))
    pending = False
    for rep in range(a.reps):
        # rep 0 reproduces the diagnostics.py H1 partition (seed + 7)
        rng = np.random.RandomState(a.seed + 7 if rep == 0 else a.seed + 100 + rep)
        part = np.full(len(y), -1)
        for s in SPECIES:
            idx = rng.permutation(np.where(src == s)[0])
            for k, ch in enumerate(np.array_split(idx, 3)):
                part[ch] = k
        for k, f in enumerate(fcs):
            key = f"{rep}|{f}"
            if key in out["auc"]:
                continue
            if left() < (110 if rep == 0 else 70):
                pending = True
                break
            te = ((y == 0) & (fc == f)) | ((y == 1) & (part == k))
            auc, acc, pr = {}, {}, {}
            for name, A in sets.items():
                p = fit_predict(A[~te], y[~te], A[te])
                auc[name] = float(roc_auc_score(y[te], p))
                acc[name] = float(accuracy_score(y[te], p >= 0.5))
                pr[name] = p
            if rep == 0:
                out["boot"][key] = {f"{x} vs {z}": paired_boot(y[te], pr[x], pr[z]) for x, z in comps}
            out["auc"][key], out["acc"][key] = auc, acc
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(out, open(a.out, "w"))
            print(key, f"stats {auc['stats']:.4f} designed {auc['stats+filters']:.4f} "
                       f"random0 {auc['stats+random0']:.4f}", flush=True)
        if pending:
            break

    if not pending:
        keys = list(out["auc"])
        summ = {}
        for name in sets:
            v = np.array([out["auc"][k][name] for k in keys])
            w = np.array([out["acc"][k][name] for k in keys])
            summ[name] = {"auc_mean": float(v.mean()), "auc_sd": float(v.std()),
                          "acc_mean": float(w.mean()), "acc_sd": float(w.std()), "n_eval": len(v)}
        pc = {}
        for x, z in comps:
            d = np.array([out["auc"][k][x] - out["auc"][k][z] for k in keys])
            pc[f"{x} vs {z}"] = {"mean_auc_diff": float(d.mean()), "sd": float(d.std()),
                                 "n_eval": len(d), "n_positive": int((d > 0).sum()),
                                 "boot95_partition0": {k: out["boot"][k][f"{x} vs {z}"]
                                                       for k in out["boot"]}}
        out["summary"], out["paired"] = summ, pc
        json.dump(out, open(a.out, "w"), indent=1)
        print("DONE")
    sys.exit(3 if pending else 0)


if __name__ == "__main__":
    main()
