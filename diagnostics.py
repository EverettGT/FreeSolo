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


CLF = {"type": "logreg"}   # set from --clf in main()


def mlp(seed):
    c = FreeSoloClassifier(model_type=CLF["type"], random_state=seed, verbose=False)
    return c


def fit_predict_features(F_tr, y_tr, F_te, seed):
    """Train scaler + classifier on precomputed features (same config as FreeSolo)."""
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
    ap.add_argument("--n-main-human", type=int, default=12000,
                    help="H: reads for the main (standard-window) evaluation")
    ap.add_argument("--n-main-per-species", type=int, default=3000)
    ap.add_argument("--n-length-survey", type=int, default=20000,
                    help="reads per class surveyed for read-length distributions")
    ap.add_argument("--clf", choices=["logreg", "mlp"], default="logreg",
                    help="classifier on the 280 features; model caches are kept per classifier")
    ap.add_argument("--budget", type=float, default=None,
                    help="stop before this many seconds; rerun to resume")
    a = ap.parse_args()
    t_start = time.time()
    CLF["type"] = a.clf
    sfx = "" if a.clf == "mlp" else f"_{a.clf}"   # MLP caches keep their original names
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
            fn = os.path.join(cache, f"model_{o}{sfx}.pkl")
            if os.path.exists(fn):
                return pickle.load(open(fn, "rb"))
            need(45)
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
            need(45)
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
                need(100)
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
                need(100)
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
            clf = FreeSoloClassifier(model_type=a.clf, verbose=False)
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
                "features_plus_classifier_mean": float(np.mean(lat_core)),
                "end_to_end_mean": float(np.mean(lat_full)),
                "end_to_end_p95": float(np.percentile(lat_full, 95)),
                "acquisition_s_single_chunk": (1500 + WINDOW) / SAMPLE_RATE,
                "hardware": f"{os.cpu_count()} CPU threads (diagnostics host)", "classifier": a.clf,
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

        # ------------------------------------------- H main evaluation
        # Standard preprocessing (one 4,000-sample window after the 1,500-sample
        # skip, no long-read restriction), larger sample, every human flow cell
        # held out in turn, species holdout with an unseen human flow cell, and
        # the ReadUntil simulation with unrestricted read lengths.
        mains = ["human"] + list(SPECIES)
        for g in mains:
            fn = os.path.join(cache, f"main_{g}.npz")
            if os.path.exists(fn):
                continue
            need(120)
            t0 = time.time()
            if g == "human":
                w, m, rej = load_human_reads(os.path.join(a.data, "human"), a.n_main_human,
                                             a.seed + 1, (1500,), WINDOW, 0)
            else:
                w, m, rej = sample_pod5(os.path.join(a.data, "pathogen", SPECIES[g]["file"]),
                                        a.n_main_per_species, random.Random(a.seed + 1), g,
                                        (1500,), WINDOW, 0)
                w = {1500: np.stack(w[1500])}
            np.savez(fn, x=w[1500])
            json.dump({"meta": m, "rejected": rej}, open(fn.replace(".npz", ".json"), "w"))
            log(f"H loaded {g}: {len(m)} (rejected {rej}) in {time.time()-t0:.0f}s")

        hmeta, hrej = [], {}
        for g in mains:
            j = json.load(open(os.path.join(cache, f"main_{g}.json")))
            hmeta += j["meta"]
            hrej[g] = j["rejected"]
        hy = np.array([0 if m["source"] == "human" else 1 for m in hmeta])
        hfc = np.array([m["flow_cell"] for m in hmeta])
        hsrc = np.array([m["source"] for m in hmeta])

        # features in chunks so each step fits a short session
        CH = 3000
        n_chunks = int(np.ceil(len(hy) / CH))
        for i in range(n_chunks):
            fn = os.path.join(cache, f"main_feat_{i}.npy")
            if os.path.exists(fn):
                continue
            need(5e-3 * CH + 10)
            X = np.concatenate([np.load(os.path.join(cache, f"main_{g}.npz"))["x"] for g in mains])
            np.save(fn, fe.extract_features(X[i * CH:(i + 1) * CH]))
            del X
            log(f"H features chunk {i+1}/{n_chunks}")
        HF = np.concatenate([np.load(os.path.join(cache, f"main_feat_{i}.npy")) for i in range(n_chunks)])

        if "H_main" not in R:
            save_read_table(hmeta, os.path.join(a.out, "reads_main.csv"))
            R["H_main"] = {"data": {
                "n_human": int((hy == 0).sum()), "n_pathogen": int((hy == 1).sum()),
                "human_per_flowcell": {f: int(((hfc == f) & (hy == 0)).sum()) for f in sorted(set(hfc[hy == 0]))},
                "species_counts": {s_: int((hsrc == s_).sum()) for s_ in SPECIES},
                "rejected": hrej}}
            dump()
        H = R["H_main"]

        # H1: each human flow cell held out once; pathogen reads split into
        # three species-stratified parts, one part tested per fold
        rng_h = np.random.RandomState(a.seed + 7)
        part = np.full(len(hy), -1)
        for s_ in SPECIES:
            idx = rng_h.permutation(np.where(hsrc == s_)[0])
            for k, chunk in enumerate(np.array_split(idx, 3)):
                part[chunk] = k
        human_fcs = sorted(set(hfc[hy == 0]))
        H.setdefault("H1_flowcell_folds", {})
        for k, f in enumerate(human_fcs):
            if f in H["H1_flowcell_folds"]:
                continue
            need(90)
            te_h = ((hy == 0) & (hfc == f)) | ((hy == 1) & (part == k))
            p, mdl = fit_predict_features(HF[~te_h], hy[~te_h], HF[te_h], a.seed)
            H["H1_flowcell_folds"][f] = summarize(hy[te_h], p)
            if f == a.test_flowcell:
                pickle.dump((te_h, p, mdl), open(os.path.join(cache, f"main_model{sfx}.pkl"), "wb"))
            log(f"H1 held-out human flow cell {f}: AUC {H['H1_flowcell_folds'][f]['auc']:.4f} "
                f"acc {H['H1_flowcell_folds'][f]['accuracy']:.4f}")
            dump()
        if "H1_mean" not in H and len(H["H1_flowcell_folds"]) == len(human_fcs):
            v = H["H1_flowcell_folds"].values()
            H["H1_mean"] = {m_: [float(np.mean([d[m_] for d in v])), float(np.std([d[m_] for d in v]))]
                            for m_ in ("auc", "accuracy", "recall_pos", "recall_neg")}
            dump()

        # H2: leave one species out AND test humans from an unseen flow cell
        H.setdefault("H2_species_holdout", {})
        for s_ in SPECIES:
            if s_ in H["H2_species_holdout"]:
                continue
            need(90)
            te_h = ((hy == 0) & (hfc == a.test_flowcell)) | (hsrc == s_)
            p, _ = fit_predict_features(HF[~te_h], hy[~te_h], HF[te_h], a.seed)
            H["H2_species_holdout"][s_] = summarize(hy[te_h], p)
            log(f"H2 held-out {s_} (+ unseen human flow cell): AUC {H['H2_species_holdout'][s_]['auc']:.4f} "
                f"pathogen recall {H['H2_species_holdout'][s_]['recall_pos']:.3f}")
            dump()

        # H4: calibration of P(pathogen) on the held-out test flow cell fold
        # (the simulator thresholds these probabilities)
        if "H4_calibration" not in H and os.path.exists(os.path.join(cache, f"main_model{sfx}.pkl")):
            te_h, p, _ = pickle.load(open(os.path.join(cache, f"main_model{sfx}.pkl"), "rb"))
            yt = hy[te_h]
            bins = np.linspace(0, 1, 11)
            rel = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m_ = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
                if m_.sum():
                    rel.append({"bin": [float(lo), float(hi)], "n": int(m_.sum()),
                                "mean_pred": float(p[m_].mean()), "frac_pathogen": float(yt[m_].mean())})
            ece = sum(r_["n"] * abs(r_["mean_pred"] - r_["frac_pathogen"]) for r_ in rel) / len(yt)
            H["H4_calibration"] = {"brier": float(np.mean((p - yt) ** 2)), "ece": float(ece),
                                   "reliability": rel, "test_flowcell": a.test_flowcell}
            log(f"H4 calibration: Brier {H['H4_calibration']['brier']:.4f} ECE {ece:.4f}")
            dump()

        # H3: read-length survey (all reads, including those too short to classify)
        fn = os.path.join(cache, "lengths.json")
        if not os.path.exists(fn):
            need(120)
            import pod5, glob
            lens = {"human": [], "pathogen": []}
            hfiles = sorted(glob.glob(os.path.join(a.data, "human", "*.pod5")))
            per = a.n_length_survey // len(hfiles)
            for fp in hfiles:
                with pod5.Reader(fp) as rd:
                    for i_, rec in enumerate(rd.reads()):
                        if i_ >= per:
                            break
                        lens["human"].append(int(rec.num_samples))
            per = a.n_length_survey // len(SPECIES)
            for s_ in SPECIES:
                with pod5.Reader(os.path.join(a.data, "pathogen", SPECIES[s_]["file"])) as rd:
                    for i_, rec in enumerate(rd.reads()):
                        if i_ >= per:
                            break
                        lens["pathogen"].append(int(rec.num_samples))
            json.dump(lens, open(fn, "w"))
            log("H3 length survey done")
        lens = json.load(open(fn))
        if "H3_simulation" not in H:
            need(150)
            te_h, p, _ = pickle.load(open(os.path.join(cache, f"main_model{sfx}.pkl"), "rb"))
            lh, lp = np.array(lens["human"], float), np.array(lens["pathogen"], float)
            lat = R["F_latency_ms"]["end_to_end_mean"]
            sim = ReadUntilSimulator(n_pores=512, human_fraction=0.99, seed=a.seed)
            yt = hy[te_h]
            H["H3_simulation"] = {
                "length_survey": {k_: {"n": len(v_), "median_bases": float(np.median(v_) / 12.5),
                                       "mean_bases": float(np.mean(v_) / 12.5),
                                       "frac_too_short_to_classify": float(np.mean(np.array(v_) < 1500 + WINDOW))}
                                  for k_, v_ in lens.items()},
                "empirical_lengths": sim.run_simulation_from_probs(
                    p[yt == 0], p[yt == 1], lh, lp, 1500 + WINDOW, lat,
                    duration_seconds=a.sim_duration, verbose=False),
                "equal_lengths(human distribution for both)": sim.run_simulation_from_probs(
                    p[yt == 0], p[yt == 1], lh, lh, 1500 + WINDOW, lat,
                    duration_seconds=a.sim_duration, verbose=False),
            }
            for k_ in ("empirical_lengths", "equal_lengths(human distribution for both)"):
                log(f"H3 {k_}: {H['H3_simulation'][k_]['enrichment']}")
            dump()

        if "H3_sensitivity" not in H:
            need(150)
            te_h, p, _ = pickle.load(open(os.path.join(cache, f"main_model{sfx}.pkl"), "rb"))
            lh, lp = np.array(lens["human"], float), np.array(lens["pathogen"], float)
            lat = R["F_latency_ms"]["end_to_end_mean"]
            yt = hy[te_h]
            grid = []
            for cap in (0.5, 2.0, 5.0):
                for unb in (0.1, 0.5, 1.0):
                    for lens_mode in ("empirical", "equal"):
                        sim = ReadUntilSimulator(n_pores=512, human_fraction=0.99, seed=a.seed,
                                                 capture_s=cap, unblock_s=unb)
                        r_ = sim.run_simulation_from_probs(
                            p[yt == 0], p[yt == 1], lh, lp if lens_mode == "empirical" else lh,
                            1500 + WINDOW, lat, duration_seconds=a.sim_duration, verbose=False)
                        grid.append({"capture_s": cap, "unblock_s": unb, "pathogen_lengths": lens_mode,
                                     **r_["enrichment"],
                                     "adaptive_pathogen_share": r_["bases"]["adaptive_pathogen_share"],
                                     "control_pathogen_share": r_["bases"]["control_pathogen_share"]})
            H["H3_sensitivity"] = grid
            log("H3 sensitivity grid done")
            dump()
        log("ALL DONE")
    except OutOfTime:
        dump()
        log("time budget reached; rerun the same command to continue")
        sys.exit(3)


if __name__ == "__main__":
    main()
