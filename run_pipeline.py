"""
FreeSolo - Master Pipeline (Real Data, 1:1:1:1 equal pathogen species ratio)

Usage:
  python run_pipeline.py
  python run_pipeline.py --n-human 50000 --n-pathogen 50000 --skip-sim

"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from real_data_loader import load_dataset, get_default_paths
from classifier import VSIClassifier
from readuntil_sim import ReadUntilSimulator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier


def parse_args():
    p = argparse.ArgumentParser(description="FreeSolo real-data pipeline")
    p.add_argument("--human-dir",    default=None,
                   help="Directory with human POD5 files")
    p.add_argument("--pathogen-dir", default=None,
                   help="Directory with per-species POD5 files")
    p.add_argument("--n-human",      type=int, default=50000)
    p.add_argument("--n-pathogen",   type=int, default=50000)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--skip-sim",     action="store_true",
                   help="Skip ReadUntil simulation step")
    return p.parse_args()


def main():
    args = parse_args()

    default_human, default_pathogen = get_default_paths()
    human_dir    = args.human_dir    or default_human
    pathogen_dir = args.pathogen_dir or default_pathogen

    base        = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base, "data")
    model_dir   = os.path.join(base, "models")
    results_dir = os.path.join(base, "results")
    figures_dir = os.path.join(results_dir, "figures")
    for d in [data_dir, model_dir, results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    t_start = time.time()

    # Load real POD5 data

    print("\n" + "="*70 + "\nSTEP 1: Loading Real POD5 Data\n" + "="*70)
    print(f"  Human dir:    {human_dir}")
    print(f"  Pathogen dir: {pathogen_dir}")
    print(f"  Target:       {args.n_human:,} human + {args.n_pathogen:,} pathogen (12,500/species)")

    X, y, load_meta = load_dataset(
        human_dir, pathogen_dir,
        n_human=args.n_human, n_pathogen=args.n_pathogen,
        seed=args.seed)

    np.save(os.path.join(data_dir, "signals.npy"), X)
    np.save(os.path.join(data_dir, "labels.npy"),  y)
    with open(os.path.join(data_dir, "load_metadata.json"), "w") as f:
        json.dump(load_meta, f, indent=2)

    print(f"\n  Final dataset: {X.shape[0]:,} reads, window={X.shape[1]} samples")
    print(f"  Human:         {int(np.sum(y==0)):,}")
    print(f"  Pathogen:      {int(np.sum(y==1)):,}")

    # Train / Val / Test split  (70 / 15 / 15, stratified)

    print("\n" + "="*70 + "\nSTEP 2: Train/Val/Test Split (70/15/15)\n" + "="*70)
    X_tv,    X_test,  y_tv,    y_test  = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y)
    X_train, X_val,   y_train, y_val   = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, random_state=args.seed, stratify=y_tv)
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # Train classifier

    print("\n" + "="*70 + "\nSTEP 3: Training\n" + "="*70)
    clf = VSIClassifier(hidden_layers=(256, 128, 64), max_iter=500,
                        random_state=args.seed)
    history = clf.train(X_train, y_train, X_val, y_val)
    save_history = dict(history)
    save_history["loss_curve"] = [float(x) for x in history["loss_curve"]]
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(save_history, f, indent=2)

    # Evaluate on held-out test set

    print("\n" + "="*70 + "\nSTEP 4: Evaluation\n" + "="*70)
    metrics = clf.evaluate(X_test, y_test)
    save_m = {k: v for k, v in metrics.items() if k != "classification_report"}
    save_m["classification_report_text"] = metrics["classification_report"]
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(save_m, f, indent=2, default=str)
    clf.save(os.path.join(model_dir, "freesolo_classifier.pkl"))

    # 10-fold cross-validation (10k stratified subsample)

    print("\n" + "="*70 + "\nSTEP 5: 10-Fold Cross-Validation\n" + "="*70)
    cv_idx    = np.random.RandomState(args.seed).choice(
        len(y), size=min(10000, len(y)), replace=False)
    X_cv_feat = clf.feature_extractor.fit_transform(X[cv_idx])
    y_cv      = y[cv_idx]
    cv_model  = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        solver="adam", alpha=1e-4, max_iter=300,
        random_state=args.seed, early_stopping=True)
    cv_scores = cross_val_score(cv_model, X_cv_feat, y_cv, cv=10, scoring="accuracy")
    cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))
    ci_lo = cv_mean - 1.96 * cv_std / np.sqrt(10)
    ci_hi = cv_mean + 1.96 * cv_std / np.sqrt(10)
    print(f"  10-Fold CV: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  95% CI:     [{ci_lo:.4f}, {ci_hi:.4f}]")
    with open(os.path.join(results_dir, "cross_validation.json"), "w") as f:
        json.dump({"cv_scores": cv_scores.tolist(), "cv_mean": cv_mean,
                   "cv_std": cv_std, "ci_95_lower": ci_lo,
                   "ci_95_upper": ci_hi}, f, indent=2)

    # ReadUntil simulation on real held-out signals

    enrichment_factor = None
    if not args.skip_sim:
        print("\n" + "="*70 + "\nSTEP 6: ReadUntil Simulation (real signals)\n" + "="*70)
        simulator = ReadUntilSimulator(
            classifier=clf, data_generator=None,
            n_pores=256, human_fraction=0.99)
        sim_results = simulator.run_simulation_from_arrays(
            X_test, y_test, duration_seconds=10)
        simulator.save_results(sim_results, results_dir)
        enrichment_factor = sim_results["enrichment"]["enrichment_factor"]
    else:
        print("\nSTEP 6: Skipped (--skip-sim)")


    # Summary

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
        print(f"  Enrichment: {enrichment_factor:.1f}x")

    with open(os.path.join(results_dir, "pipeline_summary.json"), "w") as f:
        json.dump({
            "accuracy":         metrics["accuracy"],
            "precision":        metrics["precision"],
            "recall":           metrics["recall"],
            "f1":               metrics["f1"],
            "auc_roc":          metrics["auc_roc"],
            "ci_95":            [ci_lo, ci_hi],
            "latency_ms":       metrics["latency"]["mean_ms"],
            "enrichment":       enrichment_factor,
            "n_human":          int(np.sum(y==0)),
            "n_pathogen":       int(np.sum(y==1)),
            "species_ratio":    load_meta["species_ratio"],
            "pipeline_seconds": total_time,
            "human_source":     load_meta["human_source"],
            "pathogen_source":  load_meta["pathogen_source"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
