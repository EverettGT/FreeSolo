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
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data_loader import (load_human_reads, load_pathogen_reads, save_read_table,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/nanopore"))
    ap.add_argument("--out", default="results/diagnostics")
    ap.add_argument("--n-human", type=int, default=4000)
    ap.add_argument("--n-per-species", type=int, default=1000)
    ap.add_argument("--test-flowcell", default="PAO98633")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-duration", type=int, default=3600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"params": vars(a), "offsets": OFFSETS, "window": WINDOW}
    min_len = max(OFFSETS) + WINDOW

    def dump():
        with open(os.path.join(a.out, "diagnostics.json"), "w") as f:
            json.dump(R, f, indent=2)

    # ---------------------------------------------------------------- load
    log(f"Loading human reads (min raw length {min_len} samples, all offsets)")
    hw, hmeta, hrej = load_human_reads(os.path.join(a.data, "human"), a.n_human, a.seed,
                                       OFFSETS, WINDOW, min_len)
    log(f"  human {len(hmeta)} (rejected {hrej})")
    log("Loading pathogen reads")
    pw, pmeta, prej, counts = load_pathogen_reads(os.path.join(a.data, "pathogen"),
                                                  a.n_per_species, a.seed, OFFSETS, WINDOW, min_len)
    log(f"  pathogen {counts} (rejected {prej})")
    meta = hmeta + pmeta
    save_read_table(meta, os.path.join(a.out, "reads.csv"))
    y = np.array([0] * len(hmeta) + [1] * len(pmeta))
    fc = np.array([m["flow_cell"] for m in meta])
    run = np.array([m["run_id"] for m in meta])
    src = np.array([m["source"] for m in meta])
    raw_len = np.array([m["raw_len"] for m in meta])
    R["data"] = {
        "n_human": len(hmeta), "n_pathogen": len(pmeta), "species_counts": counts,
        "human_runs": {f"{f}/{r}": int(((fc == f) & (run == r)).sum())
                       for f, r in sorted(set(zip(fc[y == 0], run[y == 0])))},
        "pathogen_flowcells": sorted(set(fc[y == 1])),
        "kits": sorted(set(m["kit"] for m in meta)),
        "devices": sorted(set(m["device"] for m in meta)),
        "median_raw_len_bases": {"human": float(np.median(raw_len[y == 0]) / 12.5),
                                 "pathogen": float(np.median(raw_len[y == 1]) / 12.5)},
    }
    dump()

    # ------------------------------------------------------------ features
    fe = Conv1DFeatureExtractor()
    fe.verbose = False
    F = {}
    for o in OFFSETS:
        t0 = time.time()
        X = np.concatenate([hw[o], pw[o]])
        F[o] = fe.extract_features(X)
        log(f"Features at offset {o}: {F[o].shape} in {time.time()-t0:.0f}s")
    del hw, pw

    # --------------------------------------------------------- split masks
    rng = np.random.RandomState(a.seed)
    is_test = np.zeros(len(y), bool)
    is_test[(y == 0) & (fc == a.test_flowcell)] = True
    p_idx = np.where(y == 1)[0]
    n_pt = int(is_test.sum())
    for sp in SPECIES:                        # stratify pathogen test by species
        sp_idx = rng.permutation(np.where(src == sp)[0])
        is_test[sp_idx[: n_pt // len(SPECIES)]] = True
    tr, te = ~is_test, is_test
    # read-level split for comparison (ignores flow cells)
    rl_te = np.zeros(len(y), bool)
    rl_te[rng.permutation(len(y))[: int(0.3 * len(y))]] = True
    R["split"] = {"test_flowcell": a.test_flowcell, "n_train": int(tr.sum()),
                  "n_test": int(te.sum()),
                  "train_human_flowcells": sorted(set(fc[tr & (y == 0)])),
                  "note": "pathogen reads all come from one flow cell; only the human side can be grouped"}

    # ------------------------------------------------- A + D offset sweep
    log("A/D: offset sweep (flow-cell-held-out split) + read-level split at 1,500")
    R["A_offset_sweep"] = {}
    models = {}
    for o in OFFSETS:
        p, models[o] = fit_predict_features(F[o][tr], y[tr], F[o][te], a.seed)
        R["A_offset_sweep"][str(o)] = summarize(y[te], p)
        log(f"  offset {o:>5}: AUC {R['A_offset_sweep'][str(o)]['auc']:.4f}")
        dump()
    p_rl, _ = fit_predict_features(F[1500][~rl_te], y[~rl_te], F[1500][rl_te], a.seed)
    R["D_read_level_split_offset1500"] = summarize(y[rl_te], p_rl)
    log(f"  read-level split @1500: AUC {R['D_read_level_split_offset1500']['auc']:.4f}")
    dump()

    # ----------------------------------------------- B run / device probes
    log("B: batch probes on human reads only")
    R["B_batch_probes"] = {}
    hmask = y == 0
    runs = sorted(set(zip(fc[hmask], run[hmask])))
    same_fc = [(f, r) for f, r in runs if f == "PAO89685"]
    probes = {}
    if len(same_fc) == 2:
        probes["same_flowcell_same_sample_two_runs (PAO89685)"] = (
            (fc == "PAO89685") & (run == same_fc[0][1]), (fc == "PAO89685") & (run == same_fc[1][1]))
    probes["PromethION24_PAO98397_vs_PromethION48_PAO98633"] = (fc == "PAO98397", fc == "PAO98633")
    probes["PAO89685_vs_PAO98633 (same instrument model)"] = (fc == "PAO89685", fc == "PAO98633")
    for name, (m0, m1) in probes.items():
        sel = m0 | m1
        if m0.sum() < 50 or m1.sum() < 50:
            continue
        yy = m1[sel].astype(int)
        R["B_batch_probes"][name] = {str(o): cv_auc(F[o][sel], yy, a.seed) for o in (1500, 9500)}
        log(f"  {name}: AUC@1500 {R['B_batch_probes'][name]['1500']['auc']:.3f}  "
            f"AUC@9500 {R['B_batch_probes'][name]['9500']['auc']:.3f}")
        dump()

    # -------------------------------------------- C positive control
    log("C: species-vs-species on the single pathogen flow cell")
    R["C_positive_control"] = {}
    pairs = [("saureus", "paeruginosa"), ("saureus", "ecoli"), ("ecoli", "kpneumoniae"),
             ("paeruginosa", "kpneumoniae")]
    for s0, s1 in pairs:
        sel = (src == s0) | (src == s1)
        yy = (src[sel] == s1).astype(int)
        R["C_positive_control"][f"{s0}_vs_{s1}"] = {str(o): cv_auc(F[o][sel], yy, a.seed) for o in (1500, 9500)}
        log(f"  {s0} vs {s1}: AUC@1500 {R['C_positive_control'][f'{s0}_vs_{s1}']['1500']['auc']:.3f}  "
            f"AUC@9500 {R['C_positive_control'][f'{s0}_vs_{s1}']['9500']['auc']:.3f}")
        dump()

    # -------------------------------------------- E multi-chunk
    log("E: multi-chunk decisions (model trained on offset-1500 chunk)")
    sc, net = models[1500]
    chunk_p = np.stack([net.predict_proba(sc.transform(F[o][te]))[:, 1] for o in CHUNK_OFFSETS], 1)
    R["E_multi_chunk"] = {}
    for k in (1, 2, 3):
        p = chunk_p[:, :k].mean(1)
        d = summarize(y[te], p)
        d["signal_needed_samples"] = CHUNK_OFFSETS[0] + k * WINDOW
        d["acquisition_time_s"] = d["signal_needed_samples"] / SAMPLE_RATE
        d["approx_bases_before_decision"] = d["signal_needed_samples"] / 12.5
        R["E_multi_chunk"][str(k)] = d
        log(f"  {k} chunk(s): AUC {d['auc']:.4f} at {d['acquisition_time_s']:.1f}s of signal")
    dump()

    # -------------------------------------------- F end-to-end latency
    log("F: end-to-end latency on raw read prefixes")
    import pod5, glob
    clf = FreeSoloClassifier(verbose=False)
    clf.feature_extractor.scaler = sc
    clf.feature_extractor.verbose = False
    clf.model = net
    lat_full, lat_pre = [], []
    f0 = sorted(glob.glob(os.path.join(a.data, "human", "*.pod5")))[0]
    with pod5.Reader(f0) as rd:
        for i, rec in enumerate(rd.reads()):
            if i >= 300:
                break
            raw = rec.signal_pa[:WINDOW + 1500]
            t0 = time.perf_counter()
            _preprocess_read(raw)
            lat_pre.append((time.perf_counter() - t0) * 1000)
            _, _, ms = clf.predict_raw(raw)
            lat_full.append(ms)
    R["F_latency_ms"] = {
        "preprocess_mean": float(np.mean(lat_pre)),
        "end_to_end_mean": float(np.mean(lat_full)), "end_to_end_p95": float(np.percentile(lat_full, 95)),
        "acquisition_s_single_chunk": (1500 + WINDOW) / SAMPLE_RATE,
        "hardware": f"{os.cpu_count()} CPU threads (diagnostics host)",
    }
    log(f"  end-to-end {R['F_latency_ms']['end_to_end_mean']:.2f} ms mean")
    dump()

    # -------------------------------------------- G simulation
    log("G: ReadUntil simulation (control vs adaptive, base-level enrichment)")
    p1 = net.predict_proba(sc.transform(F[1500][te]))[:, 1]
    lat = R["F_latency_ms"]["end_to_end_mean"]
    sim = ReadUntilSimulator(n_pores=512, human_fraction=0.99, seed=a.seed)
    # full empirical length distributions (all loaded reads, not just long ones,
    # are not available here because diagnostics require long reads; use the
    # pipeline's reads.csv for the main simulation)
    lh, lp = raw_len[y == 0], raw_len[y == 1]
    R["G_simulation"] = {
        "empirical_lengths": sim.run_simulation_from_probs(
            p1[y[te] == 0], p1[y[te] == 1], lh, lp, 1500 + WINDOW, lat,
            duration_seconds=a.sim_duration, verbose=False),
        "shared_lengths(human dist for both)": sim.run_simulation_from_probs(
            p1[y[te] == 0], p1[y[te] == 1], lh, lh, 1500 + WINDOW, lat,
            duration_seconds=a.sim_duration, verbose=False),
        "note": "diagnostic reads are restricted to >= 13,500 samples; main-pipeline simulation uses unrestricted lengths",
    }
    dump()
    log("Done")


if __name__ == "__main__":
    main()
