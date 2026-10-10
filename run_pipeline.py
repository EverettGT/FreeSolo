"""
FreeSolo - Master Pipeline

Usage:
  python run_pipeline.py
  python run_pipeline.py --n-human 50000 --n-pathogen 50000
  python run_pipeline.py --skip-sim
  python run_pipeline.py --skip-holdout
  python run_pipeline.py --split read        # original read-level split

Revision notes (Oct 2026):
  * default split holds out one human flow cell for testing (--split flowcell);
    the pathogen class comes from a single flow cell and cannot be grouped,
    which is a limitation reported in the paper
  * per-read provenance (read ID, flow cell, run, kit) saved to results/reads.csv
  * deterministic fold seeds (the old code used Python's salted hash())
  * cross-validation uses the same MLP configuration as the reported model
  * ReadUntil simulation uses empirical read lengths and a real control run
  * for confound checks see diagnostics.py
"""

import sys, os, json, time, argparse
import numpy as np
import glob
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_data_loader import (load_dataset, load_pod5_reads, load_human_reads as _load_human,
                              get_default_paths, SPECIES, save_read_table)
from classifier import FreeSoloClassifier
from readuntil_sim import ReadUntilSimulator
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPECIES_DISPLAY = {
    "saureus":     "S. aureus",
    "ecoli":       "E. coli",
    "paeruginosa": "P. aeruginosa",
    "kpneumoniae": "K. pneumoniae",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--human-dir",              default=None)
    p.add_argument("--pathogen-dir",           default=None)
    p.add_argument("--n-human",                type=int, default=50000)
    p.add_argument("--n-pathogen",             type=int, default=50000)
    p.add_argument("--n-pathogen-per-species", type=int, default=12500)
    p.add_argument("--seed",                   type=int, default=42)
    p.add_argument("--skip-sim",               action="store_true")
    p.add_argument("--skip-holdout",           action="store_true")
    p.add_argument("--split", choices=["flowcell", "read"], default="flowcell",
                   help="flowcell: test human reads come from a held-out flow cell")
    p.add_argument("--test-flowcell", default="PAO98633",
                   help="human flow cell held out for testing (default: HG004 run)")
    p.add_argument("--sim-duration", type=int, default=3600)
    p.add_argument("--model", choices=["logreg", "mlp"], default="logreg",
                   help="classifier on the 280 features (default logistic regression)")
    p.add_argument("--out-dir", default=None,
                   help="where data/, models/ and results/ are written (default: repository folder)")
    return p.parse_args()


def load_human_reads(human_pod5_dir, n_human, seed):
    """Equal per-file quotas; returns (X, provenance list)."""
    wins, meta, rej = _load_human(human_pod5_dir, n_human, seed)
    X_h = wins[1500]
    print(f"  Loaded: {len(X_h):,}  Rejected: {rej:,}")
    return X_h, meta


def run_holdout_fold(held_out, X_human, species_data, seed, fold_dir,
                     human_flowcells=None, test_flowcell=None, model_type="logreg"):
    display = SPECIES_DISPLAY[held_out]
    print(f"\n{'='*70}\nFOLD: Hold out {display}\n{'='*70}")

    # deterministic per-fold seed (hash() of a str is salted per process)
    fold_seed = seed + 1000 * (list(SPECIES).index(held_out) + 1)
    rng = np.random.RandomState(fold_seed)

    if human_flowcells is not None and test_flowcell is not None:
        # human test reads come from a flow cell never seen in training
        fc = np.asarray(human_flowcells)
        test_human_idx  = rng.permutation(np.where(fc == test_flowcell)[0])
        train_human_idx = np.where(fc != test_flowcell)[0]
        n_test_human = len(test_human_idx)
    else:
        n_test_human = int(len(X_human) * 0.15)
        all_idx = rng.permutation(len(X_human))
        test_human_idx  = all_idx[:n_test_human]
        train_human_idx = all_idx[n_test_human:]

    assert len(np.intersect1d(train_human_idx, test_human_idx)) == 0

    X_train_parts = [X_human[train_human_idx]]
    y_train_parts = [np.zeros(len(train_human_idx), dtype=np.int32)]

    for name, X_sp in species_data.items():
        if name != held_out:
            X_train_parts.append(X_sp)
            y_train_parts.append(np.ones(len(X_sp), dtype=np.int32))
            print(f"  Train: {SPECIES_DISPLAY[name]:<22} {len(X_sp):,} reads")

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    shuf = rng.permutation(len(y_train))
    X_train, y_train = X_train[shuf], y_train[shuf]

    X_held = species_data[held_out]
    n_test_p = len(X_held)
    n_test_h = min(n_test_human, n_test_p)

    X_test = np.concatenate([X_human[test_human_idx[:n_test_h]], X_held])
    y_test = np.concatenate([np.zeros(n_test_h, dtype=np.int32),
                             np.ones(n_test_p,  dtype=np.int32)])
    tshuf = rng.permutation(len(y_test))
    X_test, y_test = X_test[tshuf], y_test[tshuf]

    for name, X_sp in species_data.items():
        if name != held_out:
            assert not np.shares_memory(X_held, X_sp)

    print(f"  Test:  {display:<22} {n_test_p:,} pathogen + {n_test_h:,} human")
    print(f"  Train: {len(y_train):,}  (human={int(np.sum(y_train==0)):,}  pathogen={int(np.sum(y_train==1)):,})")

    t0 = time.time()
    clf = FreeSoloClassifier(model_type=model_type, random_state=seed)
    history = clf.train(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.0f}s")

    save_history = dict(history)
    save_history["loss_curve"] = [float(x) for x in (history["loss_curve"] or [])]
    with open(os.path.join(fold_dir, "training_history.json"), "w") as f:
        json.dump(save_history, f, indent=2)

    print(f"\n  Evaluating on held-out {display}...")
    features = clf.feature_extractor.transform(X_test)
    y_pred   = clf.model.predict(features)
    y_proba  = clf.model.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    latencies = []
    for i in range(min(100, len(X_test))):
        _, _, lat = clf.predict_single(X_test[i])
        latencies.append(lat)

    eval_metrics = {
        "accuracy":   float(accuracy_score(y_test, y_pred)),
        "precision":  float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc":    float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve":  {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "latency": {
            "mean_ms":   float(np.mean(latencies)),
            "std_ms":    float(np.std(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms":    float(np.percentile(latencies, 95)),
            "p99_ms":    float(np.percentile(latencies, 99)),
            "min_ms":    float(np.min(latencies)),
            "max_ms":    float(np.max(latencies)),
        },
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=["Human", display]
        ),
    }

    with open(os.path.join(fold_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2, default=str)

    clf.save(os.path.join(fold_dir, "models", "freesolo_classifier.pkl"))

    fold_summary = {
        "held_out_species":      held_out,
        "held_out_species_name": display,
        "accuracy":    eval_metrics["accuracy"],
        "precision":   eval_metrics["precision"],
        "recall":      eval_metrics["recall"],
        "f1":          eval_metrics["f1"],
        "auc_roc":     eval_metrics["auc_roc"],
        "latency_ms":  eval_metrics["latency"]["mean_ms"],
        "n_train":          int(len(y_train)),
        "n_train_human":    int(np.sum(y_train == 0)),
        "n_train_pathogen": int(np.sum(y_train == 1)),
        "n_test":           int(len(y_test)),
        "n_test_human":     int(n_test_h),
        "n_test_pathogen":  int(n_test_p),
        "train_time_seconds": float(train_time),
        "fold_seed":   int(fold_seed),
    }

    with open(os.path.join(fold_dir, "pipeline_summary.json"), "w") as f:
        json.dump(fold_summary, f, indent=2)

    print(f"\n  Accuracy:  {eval_metrics['accuracy']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall:    {eval_metrics['recall']:.4f}")
    print(f"  F1:        {eval_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {eval_metrics['auc_roc']:.4f}")
    print(f"\n{eval_metrics['classification_report_text']}")

    return fold_summary


def main():
    args = parse_args()

    default_human, default_pathogen = get_default_paths()
    human_dir    = args.human_dir    or default_human
    pathogen_dir = args.pathogen_dir or default_pathogen

    base         = os.path.dirname(os.path.abspath(__file__))
    base         = args.out_dir or base
    data_dir     = os.path.join(base, "data")
    model_dir    = os.path.join(base, "models")
    results_dir  = os.path.join(base, "results")
    holdout_dir  = os.path.join(results_dir, "holdout_species")

    for d in [data_dir, model_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    t_start = time.time()

    print("\n" + "="*70 + "\nSTEP 1: Loading Real POD5 Data\n" + "="*70)
    print(f"  Human dir:    {human_dir}")
    print(f"  Pathogen dir: {pathogen_dir}")

    X, y, load_meta = load_dataset(
        human_dir, pathogen_dir,
        n_human=args.n_human, n_pathogen=args.n_pathogen,
        seed=args.seed)

    reads = load_meta.pop("reads")
    np.save(os.path.join(data_dir, "signals.npy"), X)
    np.save(os.path.join(data_dir, "labels.npy"),  y)
    with open(os.path.join(data_dir, "load_metadata.json"), "w") as f:
        json.dump(load_meta, f, indent=2)
    save_read_table(reads, os.path.join(results_dir, "reads.csv"))
    flowcell = np.array([r["flow_cell"] for r in reads])
    raw_len  = np.array([r["raw_len"] for r in reads])

    print(f"\n  Final dataset: {X.shape[0]:,} reads, window={X.shape[1]}")
    print(f"  Human:         {int(np.sum(y==0)):,}")
    print(f"  Pathogen:      {int(np.sum(y==1)):,}")

    idx_all = np.arange(len(y))
    if args.split == "flowcell":
        print("\n" + "="*70 + f"\nSTEP 2: Split (human test = held-out flow cell {args.test_flowcell})\n" + "="*70)
        h_test = idx_all[(y == 0) & (flowcell == args.test_flowcell)]
        h_rest = idx_all[(y == 0) & (flowcell != args.test_flowcell)]
        p_all  = np.random.RandomState(args.seed).permutation(idx_all[y == 1])
        n_pt   = len(h_test)
        p_test, p_rest = p_all[:n_pt], p_all[n_pt:]
        test_idx = np.concatenate([h_test, p_test])
        rest_idx = np.concatenate([h_rest, p_rest])
        tr_idx, va_idx = train_test_split(rest_idx, test_size=0.15/0.85,
                                          random_state=args.seed, stratify=y[rest_idx])
    else:
        print("\n" + "="*70 + "\nSTEP 2: Train/Val/Test Split (70/15/15, read-level)\n" + "="*70)
        tv_idx, test_idx = train_test_split(idx_all, test_size=0.15, random_state=args.seed, stratify=y)
        tr_idx, va_idx = train_test_split(tv_idx, test_size=0.15/0.85, random_state=args.seed, stratify=y[tv_idx])
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]
    X_test,  y_test  = X[test_idx], y[test_idx]
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    print("\n" + "="*70 + "\nSTEP 3: Training\n" + "="*70)
    clf = FreeSoloClassifier(model_type=args.model, random_state=args.seed)
    history = clf.train(X_train, y_train, X_val, y_val)
    save_history = dict(history)
    save_history["loss_curve"] = [float(x) for x in (history["loss_curve"] or [])]
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(save_history, f, indent=2)

    print("\n" + "="*70 + "\nSTEP 4: Evaluation\n" + "="*70)
    metrics = clf.evaluate(X_test, y_test)
    save_m = {k: v for k, v in metrics.items() if k != "classification_report"}
    save_m["classification_report_text"] = metrics["classification_report"]
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(save_m, f, indent=2, default=str)
    clf.save(os.path.join(model_dir, "freesolo_classifier.pkl"))

    print("\n" + "="*70 + "\nSTEP 5: 10-Fold Cross-Validation\n" + "="*70)
    cv_idx = np.random.RandomState(args.seed).choice(
        len(y_train), size=min(10000, len(y_train)), replace=False)
    X_cv_feat = clf.feature_extractor.extract_features(X_train[cv_idx])
    y_cv      = y_train[cv_idx]
    # same classifier configuration as the reported model
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000)) if args.model == "logreg" else ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", alpha=1e-4, batch_size=64, learning_rate="adaptive",
            learning_rate_init=1e-3, max_iter=500, random_state=args.seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)),
    ])
    cv_scores = cross_val_score(cv_pipeline, X_cv_feat, y_cv, scoring="accuracy",
                                cv=StratifiedKFold(10, shuffle=True, random_state=args.seed))
    cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))
    ci_lo = cv_mean - 1.96 * cv_std / np.sqrt(10)
    ci_hi = cv_mean + 1.96 * cv_std / np.sqrt(10)
    print(f"  10-Fold CV: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  95% CI:     [{ci_lo:.4f}, {ci_hi:.4f}]")
    with open(os.path.join(results_dir, "cross_validation.json"), "w") as f:
        json.dump({"cv_scores": cv_scores.tolist(), "cv_mean": cv_mean,
                   "cv_std": cv_std, "ci_95_lower": ci_lo,
                   "ci_95_upper": ci_hi}, f, indent=2)

    enrichment_factor = None
    if not args.skip_sim:
        print("\n" + "="*70 + "\nSTEP 6: ReadUntil Simulation\n" + "="*70)
        simulator = ReadUntilSimulator(classifier=clf, n_pores=512, human_fraction=0.99,
                                       seed=args.seed)
        sim_results = simulator.run_simulation_from_arrays(
            X_test, y_test, duration_seconds=args.sim_duration,
            len_human=raw_len[test_idx][y_test == 0],
            len_pathogen=raw_len[test_idx][y_test == 1])
        simulator.save_results(sim_results, results_dir)
        enrichment_factor = sim_results["enrichment"]["relative_enrichment_bases"]
    else:
        print("\nSTEP 6: Skipped (--skip-sim)")

    if not args.skip_holdout:
        print("\n" + "="*70 + "\nSTEP 7: Leave-One-Species-Out Generalization\n" + "="*70)
        os.makedirs(holdout_dir, exist_ok=True)

        print("\n  Loading human reads for holdout folds...")
        X_human_ho, human_meta_ho = load_human_reads(human_dir, args.n_human, args.seed)
        human_fc_ho = [m["flow_cell"] for m in human_meta_ho] if args.split == "flowcell" else None

        print("\n  Loading per-species pathogen reads...")
        species_data = {}
        for name, spec in SPECIES.items():
            fpath = os.path.join(pathogen_dir, spec["file"])
            print(f"  {SPECIES_DISPLAY[name]}  ({fpath})")
            X_sp, _, n_rej = load_pod5_reads(
                fpath, args.n_pathogen_per_species, label=1, seed=args.seed)
            species_data[name] = X_sp
            print(f"    Loaded: {len(X_sp):,}  Rejected: {n_rej:,}")

        holdout_results = []
        for held_out in SPECIES.keys():
            fold_dir = os.path.join(holdout_dir, f"fold_{held_out}")
            os.makedirs(os.path.join(fold_dir, "models"), exist_ok=True)
            result = run_holdout_fold(
                held_out, X_human_ho, species_data, args.seed, fold_dir,
                human_flowcells=human_fc_ho,
                test_flowcell=args.test_flowcell if human_fc_ho else None,
                model_type=args.model)
            holdout_results.append(result)

        aucs    = [r["auc_roc"]  for r in holdout_results]
        accs    = [r["accuracy"] for r in holdout_results]
        recalls = [r["recall"]   for r in holdout_results]

        holdout_summary = {
            "experiment":             "leave_one_species_out",
            "n_human_reads":          int(len(X_human_ho)),
            "n_pathogen_per_species": args.n_pathogen_per_species,
            "seed":                   args.seed,
            "folds":                  holdout_results,
            "aggregate": {
                "mean_auc_roc":  float(np.mean(aucs)),
                "std_auc_roc":   float(np.std(aucs)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_recall":   float(np.mean(recalls)),
                "std_recall":    float(np.std(recalls)),
            },
        }

        with open(os.path.join(holdout_dir, "holdout_summary.json"), "w") as f:
            json.dump(holdout_summary, f, indent=2)

        print(f"\n{'='*70}\nHOLDOUT SUMMARY\n{'='*70}")
        print(f"  {'Species':<24} {'AUC-ROC':>8}  {'Accuracy':>9}  {'Recall':>8}")
        print(f"  {'-'*24} {'-'*8}  {'-'*9}  {'-'*8}")
        for r in holdout_results:
            print(f"  {r['held_out_species_name']:<24} "
                  f"{r['auc_roc']:>8.4f}  {r['accuracy']:>9.4f}  {r['recall']:>8.4f}")
        print(f"  {'-'*24} {'-'*8}  {'-'*9}  {'-'*8}")
        print(f"  {'Mean':<24} {holdout_summary['aggregate']['mean_auc_roc']:>8.4f}  "
              f"{holdout_summary['aggregate']['mean_accuracy']:>9.4f}  "
              f"{holdout_summary['aggregate']['mean_recall']:>8.4f}")
    else:
        print("\nSTEP 7: Skipped (--skip-holdout)")
        holdout_summary = None

    total_time = time.time() - t_start

    print(f"\n{'='*70}\nPIPELINE COMPLETE ({total_time:.0f}s)\n{'='*70}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"  95% CI:     [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Latency:    {metrics['latency']['mean_ms']:.2f} ms (mean)")
    if enrichment_factor:
        print(f"  Relative enrichment (bases): {enrichment_factor:.2f}x")
    if holdout_summary:
        print(f"  Holdout AUC (mean): {holdout_summary['aggregate']['mean_auc_roc']:.4f}")

    pipeline_summary = {
        "accuracy":         metrics["accuracy"],
        "precision":        metrics["precision"],
        "recall":           metrics["recall"],
        "f1":               metrics["f1"],
        "auc_roc":          metrics["auc_roc"],
        "ci_95":            [ci_lo, ci_hi],
        "latency_ms":       metrics["latency"]["mean_ms"],
        "enrichment_relative_bases": enrichment_factor,
        "split":            args.split,
        "test_flowcell":    args.test_flowcell if args.split == "flowcell" else None,
        "n_human":          int(np.sum(y==0)),
        "n_pathogen":       int(np.sum(y==1)),
        "species_ratio":    load_meta["species_ratio"],
        "pipeline_seconds": total_time,
        "human_source":     load_meta["human_source"],
        "pathogen_source":  load_meta["pathogen_source"],
    }

    if holdout_summary:
        pipeline_summary["holdout_mean_auc_roc"] = holdout_summary["aggregate"]["mean_auc_roc"]
        pipeline_summary["holdout_mean_recall"]  = holdout_summary["aggregate"]["mean_recall"]

    with open(os.path.join(results_dir, "pipeline_summary.json"), "w") as f:
        json.dump(pipeline_summary, f, indent=2)


if __name__ == "__main__":
    main()
