"""
FreeSolo confound diagnostics (revision, Oct 2026)

Addresses Reviewer 2 (platform confound) and Reviewer 1 (multi-chunk,
latency). In the original data the class label is confounded with
instrument (PromethION vs GridION), library kit (SQK-LSK114 ligation vs
SQK-RBK114-96 rapid barcoding), flow cell and run. These experiments ask how
much of the human-vs-pathogen accuracy survives controls for that.

  A. Offset sweep        : same reads, windows starting at 0 / 1,500 / 3,000 /
                           5,500 / 9,500 samples. A large drop as the window
                           moves away from the read start indicates leakage
                           from adapter / barcode signal (pathogen reads carry
                           a rapid-barcode adapter; human reads do not).
  B. Run / device probe  : can the same features tell apart two human runs on
                           the SAME flow cell and sample (pure run effect), or
                           two human flow cells on different instruments?
                           High AUC means the features encode batch effects.
  C. Positive control    : species vs species on the SAME pathogen flow cell
                           (platform held constant). High AUC means the
                           features can capture organism signal.
  D. Group-aware split   : human test reads from a held-out flow cell.
  E. Multi-chunk         : average P(pathogen) over 1, 2, 3 consecutive
                           4,000-sample chunks; accuracy vs decision time.
  F. End-to-end latency  : preprocessing + features + MLP on raw prefixes.
  G. Corrected ReadUntil simulation with empirical read lengths.

Usage:
  python diagnostics.py --data ~/Downloads/nanopore --out results/diagnostics \
      --n-human 4000 --n-per-species 1000

Every step is checkpointed (loaded windows and features in --cache, results
in <out>/diagnostics.json). With --budget SECONDS the script stops cleanly
before the budget runs out and exits with code 3; rerun the same command to
continue. This lets the analysis run in short sessions on a laptop.
"""
import argparse
import json
import pickle
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
from real_data_loader import (load_human_reads, sample_pod5, save_read_table,
                              SPECIES, SAMPLE_RATE, _preprocess_read)
from classifier import FreeSoloClassifier, Conv1DFeatureExtractor
from readuntil_sim import ReadUntilSimulator

OFFSETS = (0, 1500, 3000, 5500, 9500)
WINDOW = 4000
CHUNK_OFFSETS = (1500, 5500, 9500)   # three consecutive non-overlapping chunks


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def boot_auc(y, p, n=500, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"auc": float(roc_auc_score(y, p)), "auc_ci95": boot_auc(y, p),
            "accuracy": float(accuracy_score(y, pred)),
            "recall_pos": float(recall_score(y, pred, zero_division=0)),
            "recall_neg": float(recall_score(1 - y, 1 - pred, zero_division=0)),
            "n": int(len(y))}


def mlp(seed):
    c = FreeSoloClassifier(random_state=seed, verbose=False)
    return c


def fit_predict_features(F_tr, y_tr, F_te, seed):
    """Train scaler+MLP on precomputed features (same config as FreeSolo)."""
    from sklearn.preprocessing import StandardScaler
    c = mlp(seed)
    sc = StandardScaler().fit(F_tr)
    c.model.fit(sc.transform(F_tr), y_tr)
    return c.model.predict_proba(sc.transform(F_te))[:, 1], (sc, c.model)


def cv_auc(F, y, seed, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(F, y):
        p[te], _ = fit_predict_features(F[tr], y[tr], F[te], seed)
    return summarize(y, p)


class OutOfTime(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/nanopore"))
    ap.add_argument("--out", default="results/diagnostics")
    ap.add_argument("--cache", default=None, help="checkpoint dir (default <out>/cache)")
    ap.add_argument("--n-human", type=int, default=4000)
    ap.add_argument("--n-per-species", type=int, default=1000)
    ap.add_argument("--test-flowcell", default="PAO98633")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-duration", type=int, default=3600)
    ap.add_argument("--budget", type=float, default=None,
                    help="stop before this many seconds; rerun to resume")
    a = ap.parse_args()
    t_start = time.time()
    cache = a.cache or os.path.join(a.out, "cache")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    res_path = os.path.join(a.out, "diagnostics.json")
    R = json.load(open(res_path)) if os.path.exists(res_path) else {}
    R.update({"params": {k: v for k, v in vars(a).items() if k not in ("budget", "cache")},
              "offsets": list(OFFSETS), "window": WINDOW})
    min_len = max(OFFSETS) + WINDOW

    def dump():
        with open(res_path, "w") as f:
            json.dump(R, f, indent=2)

    units_run = [0]

    def need(est_s):
        """Raise OutOfTime if the next unit (est_s seconds) may not fit.
        At least one unit always runs per invocation, so every call progresses."""
        if a.budget and units_run[0] > 0 and time.time() - t_start + est_s > a.budget:
            raise OutOfTime()
        units_run[0] += 1

    def done(*keys):
        d = R
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return False
            d = d[k]
        return True

    try:
        # ------------------------------------------------------------ load
        groups = ["human"] + list(SPECIES)
        for g in groups:
            fn = os.path.join(cache, f"win_{g}.npz")
            if os.path.exists(fn):
                continue
            need(90)
            t0 = time.time()
            if g == "human":
                w, m, rej = load_human_reads(os.path.join(a.data, "human"), a.n_human, a.seed,
                                             OFFSETS, WINDOW, min_len)
            else:
                w, m, rej = sample_pod5(os.path.join(a.data, "pathogen", SPECIES[g]["file"]),
                                        a.n_per_species, random.Random(a.seed), g,
                                        OFFSETS, WINDOW, min_len)
                w = {o: np.stack(v) for o, v in w.items()}
            np.savez(fn, **{str(o): w[o] for o in OFFSETS})
            json.dump({"meta": m, "rejected": rej}, open(fn.replace(".npz", ".json"), "w"))
            log(f"loaded {g}: {len(m)} reads (rejected {rej}) in {time.time()-t0:.0f}s")

        meta, rejected = [], {}
        for g in groups:
            j = json.load(open(os.path.join(cache, f"win_{g}.json")))
            meta += j["meta"]
            rejected[g] = j["rejected"]
        y = np.array([0 if m["source"] == "human" else 1 for m in meta])
        fc = np.array([m["flow_cell"] for m in meta])
        run = np.array([m["run_id"] for m in meta])
        src = np.array([m["source"] for m in meta])
        raw_len = np.array([m["raw_len"] for m in meta])
        if "data" not in R:
            save_read_table(meta, os.path.join(a.out, "reads.csv"))
            R["data"] = {
                "n_human": int((y == 0).sum()), "n_pathogen": int((y == 1).sum()),
                "species_counts": {s: int((src == s).sum()) for s in SPECIES},
                "rejected_short_or_qc": rejected,
                "min_raw_len_samples": min_len,
                "human_runs": {f"{f}/{r}": int(((fc == f) & (run == r)).sum())
                               for f, r in sorted(set(zip(fc[y == 0], run[y == 0])))},
                "pathogen_flowcells": sorted(set(fc[y == 1])),
                "kits": sorted(set(m["kit"] for m in meta)),
                "devices": sorted(set(m["device"] for m in meta)),
                "median_raw_len_bases": {"human": float(np.median(raw_len[y == 0]) / 12.5),
                                         "pathogen": float(np.median(raw_len[y == 1]) / 12.5)},
            }
            dump()

        # -------------------------------------------------------- features
        fe = Conv1DFeatureExtractor()
        fe.verbose = False
        F = {}
        for o in OFFSETS:
            fn = os.path.join(cache, f"feat_{o}.npy")
            if not os.path.exists(fn):
                need(5e-3 * len(y) + 10)
                t0 = time.time()
                X = np.concatenate([np.load(os.path.join(cache, f"win_{g}.npz"))[str(o)] for g in groups])
                np.save(fn, fe.extract_features(X))
                log(f"features @ offset {o}: {time.time()-t0:.0f}s")
            F[o] = np.load(fn)

        # ----------------------------------------------------- split masks
        rng = np.random.RandomState(a.seed)
        is_test = np.zeros(len(y), bool)
        is_test[(y == 0) & (fc == a.test_flowcell)] = True
        n_pt = int(is_test.sum())
        for sp in SPECIES:                       # pathogen test reads, stratified by species
            sp_idx = rng.permutation(np.where(src == sp)[0])
            is_test[sp_idx[: n_pt // len(SPECIES)]] = True
        tr, te = ~is_test, is_test
        rl_te = np.zeros(len(y), bool)           # read-level split for comparison
        rl_te[rng.permutation(len(y))[: int(0.3 * len(y))]] = True
        R["split"] = {"test_flowcell": a.test_flowcell, "n_train": int(tr.sum()),
                      "n_test": int(te.sum()),
                      "train_human_flowcells": sorted(set(fc[tr & (y == 0)])),
                      "note": "pathogen reads all come from one flow cell; only the human side can be grouped"}

        def fit(o):
            """Model trained on the flow-cell-held-out split at offset o (cached)."""
            fn = os.path.join(cache, f"model_{o}.pkl")
            if os.path.exists(fn):
                return pickle.load(open(fn, "rb"))
            need(120)
            p, m = fit_predict_features(F[o][tr], y[tr], F[o][te], a.seed)
            pickle.dump((p, m), open(fn, "wb"))
            return p, m

        # ------------------------------------------- A/D offset sweep
        R.setdefault("A_offset_sweep", {})
        for o in OFFSETS:
            if done("A_offset_sweep", str(o)):
                continue
            p, _ = fit(o)
            R["A_offset_sweep"][str(o)] = summarize(y[te], p)
            log(f"A offset {o:>5}: AUC {R['A_offset_sweep'][str(o)]['auc']:.4f}")
            dump()
        if not done("D_read_level_split_offset1500"):
            need(120)
            p_rl, _ = fit_predict_features(F[1500][~rl_te], y[~rl_te], F[1500][rl_te], a.seed)
            R["D_read_level_split_offset1500"] = summarize(y[rl_te], p_rl)
            log(f"D read-level split @1500: AUC {R['D_read_level_split_offset1500']['auc']:.4f}")
            dump()

        # ------------------------------------------- B batch probes
        R.setdefault("B_batch_probes", {})
        hmask = y == 0
        runs = sorted(set(zip(fc[hmask], run[hmask])))
        same_fc = [(f, r) for f, r in runs if f == "PAO89685"]
        probes = {}
        if len(same_fc) == 2:
            probes["same_flowcell_same_sample_two_runs (PAO89685)"] = (
                (fc == "PAO89685") & (run == same_fc[0][1]), (fc == "PAO89685") & (run == same_fc[1][1]))
        probes["PromethION24_PAO98397_vs_PromethION48_PAO98633 (HG003 vs HG004)"] = (fc == "PAO98397", fc == "PAO98633")
        probes["PAO89685_vs_PAO98633 (PromethION48, HG002 vs HG004)"] = (fc == "PAO89685", fc == "PAO98633")
        for name, (m0, m1) in probes.items():
            sel = m0 | m1
            if m0.sum() < 50 or m1.sum() < 50:
                continue
            R["B_batch_probes"].setdefault(name, {"n": [int(m0.sum()), int(m1.sum())]})
            for o in (1500, 9500):
                if done("B_batch_probes", name, str(o)):
                    continue
                need(150)
                R["B_batch_probes"][name][str(o)] = cv_auc(F[o][sel], m1[sel].astype(int), a.seed)
                log(f"B {name} @{o}: AUC {R['B_batch_probes'][name][str(o)]['auc']:.3f}")
                dump()

        # ------------------------------------------- C positive control
        R.setdefault("C_positive_control", {})
        pairs = [("saureus", "paeruginosa"), ("saureus", "ecoli"), ("ecoli", "kpneumoniae"),
                 ("paeruginosa", "kpneumoniae")]
        for s0, s1 in pairs:
            key = f"{s0}_vs_{s1}"
            sel = (src == s0) | (src == s1)
            R["C_positive_control"].setdefault(key, {})
            for o in (1500, 9500):
                if done("C_positive_control", key, str(o)):
                    continue
                need(120)
                R["C_positive_control"][key][str(o)] = cv_auc(F[o][sel], (src[sel] == s1).astype(int), a.seed)
                log(f"C {key} @{o}: AUC {R['C_positive_control'][key][str(o)]['auc']:.3f}")
                dump()

        # ------------------------------------------- E multi-chunk
        _, (sc, net) = fit(1500)
        if not done("E_multi_chunk"):
            chunk_p = np.stack([net.predict_proba(sc.transform(F[o][te]))[:, 1] for o in CHUNK_OFFSETS], 1)
            E = {}
            for k in (1, 2, 3):
                d = summarize(y[te], chunk_p[:, :k].mean(1))
                d["signal_needed_samples"] = CHUNK_OFFSETS[0] + k * WINDOW
                d["acquisition_time_s"] = d["signal_needed_samples"] / SAMPLE_RATE
                d["approx_bases_before_decision"] = d["signal_needed_samples"] / 12.5
                E[str(k)] = d
                log(f"E {k} chunk(s): AUC {d['auc']:.4f} after {d['acquisition_time_s']:.1f}s of signal")
            R["E_multi_chunk"] = E
            dump()

        # ------------------------------------------- F end-to-end latency
        if not done("F_latency_ms"):
            need(60)
            import pod5, glob
            clf = FreeSoloClassifier(verbose=False)
            clf.feature_extractor.scaler = sc
            clf.feature_extractor.verbose = False
            clf.model = net
            lat_full, lat_pre, lat_core = [], [], []
            f0 = sorted(glob.glob(os.path.join(a.data, "human", "*.pod5")))[0]
            with pod5.Reader(f0) as rd:
                for rec in rd.reads():
                    if len(lat_full) >= 300:
                        break
                    raw = rec.signal_pa[:WINDOW + 1500]
                    t0 = time.perf_counter()
                    win = _preprocess_read(raw)
                    t1 = time.perf_counter()
                    if win is None:
                        continue
                    lat_pre.append((t1 - t0) * 1000)
                    lat_core.append(clf.predict_single(win)[2])
                    lat_full.append(clf.predict_raw(raw)[2])
            R["F_latency_ms"] = {
                "n": len(lat_full),
                "preprocess_mean": float(np.mean(lat_pre)),
                "features_plus_mlp_mean": float(np.mean(lat_core)),
                "end_to_end_mean": float(np.mean(lat_full)),
                "end_to_end_p95": float(np.percentile(lat_full, 95)),
                "acquisition_s_single_chunk": (1500 + WINDOW) / SAMPLE_RATE,
                "hardware": f"{os.cpu_count()} CPU threads (diagnostics host)",
            }
            log(f"F end-to-end {R['F_latency_ms']['end_to_end_mean']:.2f} ms mean")
            dump()

        # ------------------------------------------- G simulation
        if not done("G_simulation"):
            need(120)
            p1 = net.predict_proba(sc.transform(F[1500][te]))[:, 1]
            lat = R["F_latency_ms"]["end_to_end_mean"]
            sim = ReadUntilSimulator(n_pores=512, human_fraction=0.99, seed=a.seed)
            lh, lp = raw_len[y == 0], raw_len[y == 1]
            R["G_simulation"] = {
                "empirical_lengths": sim.run_simulation_from_probs(
                    p1[y[te] == 0], p1[y[te] == 1], lh, lp, 1500 + WINDOW, lat,
                    duration_seconds=a.sim_duration, verbose=False),
                "shared_lengths(human dist for both)": sim.run_simulation_from_probs(
                    p1[y[te] == 0], p1[y[te] == 1], lh, lh, 1500 + WINDOW, lat,
                    duration_seconds=a.sim_duration, verbose=False),
                "note": "diagnostic reads are restricted to >= 13,500 samples; the main pipeline simulation uses unrestricted lengths",
            }
            dump()
        log("ALL DONE")
    except OutOfTime:
        dump()
        log("time budget reached; rerun the same command to continue")
        sys.exit(3)


if __name__ == "__main__":
    main()
