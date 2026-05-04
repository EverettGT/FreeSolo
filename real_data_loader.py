"""
Real Data Loader for FreeSolo

Loads raw nanopore signals from POD5 files with controlled per-species sampling.

Species (ENA PRJEB51164, R10.4.1 LSK114):
  saureus.pod5     - Staphylococcus aureus    (barcode10, ~15k reads)
  ecoli.pod5       - Escherichia coli         (barcode09, ~81k reads)
  paeruginosa.pod5 - Pseudomonas aeruginosa   (barcode11, ~90k reads)
  kpneumoniae.pod5 - Klebsiella pneumoniae    (barcode12, ~155k reads)

Pathogen sampling ratio: 1:1:1:1  (equal species representation)
11,500 reads per species regardless of available volume, ensuring
each of the four clinically relevant pathogens contributes equally.

Preprocessing (paper Section 2.2):
  1. Adapter skip  : discard first 1,500 samples
  2. Quality filter: mean outside [50,200] pA or std < 1 pA -> discard
  3. MAD normalise : clip outliers > 3.5 MAD, then zero-mean unit-variance
  4. Window        : 4,000-sample window after adapter skip

"""

import numpy as np
import os
import glob
import random
from pathlib import Path

# Preprocessing constants
ADAPTER_SKIP  = 1500
WINDOW_SIZE   = 4000
MEAN_MIN_PA   = 50.0
MEAN_MAX_PA   = 200.0
STD_MIN_PA    = 1.0
MAD_THRESHOLD = 3.5

# Species config
SPECIES = {
    "saureus":     {"file": "saureus.pod5",     "weight": 1},
    "ecoli":       {"file": "ecoli.pod5",        "weight": 1},
    "paeruginosa": {"file": "paeruginosa.pod5",  "weight": 1},
    "kpneumoniae": {"file": "kpneumoniae.pod5",  "weight": 1},
}
TOTAL_WEIGHT = sum(s["weight"] for s in SPECIES.values())  # 4


def _mad_normalise(signal: np.ndarray) -> np.ndarray:
    median = np.median(signal)
    mad    = np.median(np.abs(signal - median))
    if mad < 1e-6:
        mad = 1e-6
    modified_z   = 0.6745 * (signal - median) / mad
    outlier_mask = np.abs(modified_z) > MAD_THRESHOLD
    if np.any(outlier_mask):
        clean = signal.copy()
        for idx in np.where(outlier_mask)[0]:
            lo, hi     = max(0, idx - 2), min(len(signal), idx + 3)
            neighbours = signal[lo:hi][~outlier_mask[lo:hi]]
            clean[idx] = np.mean(neighbours) if len(neighbours) > 0 else median
        signal = clean
    std = np.std(signal)
    return (signal - np.mean(signal)) / (std if std > 1e-6 else 1e-6)


def _quality_ok(raw: np.ndarray) -> bool:
    m, s = np.mean(raw), np.std(raw)
    return MEAN_MIN_PA <= m <= MEAN_MAX_PA and s >= STD_MIN_PA


def _preprocess_read(raw: np.ndarray):
    if len(raw) < ADAPTER_SKIP + WINDOW_SIZE:
        return None
    window = raw[ADAPTER_SKIP: ADAPTER_SKIP + WINDOW_SIZE].astype(np.float32)
    if not _quality_ok(window):
        return None
    return _mad_normalise(window).astype(np.float32)


def load_pod5_reads(pod5_path: str, n_reads: int, label: int, seed: int = 42):
    """Load up to n_reads from a single POD5 file."""
    try:
        import pod5
    except ImportError:
        raise ImportError("pod5 not installed. Run: pip install pod5")

    rng = random.Random(seed)
    signals, n_rejected = [], 0

    with pod5.Reader(pod5_path) as reader:
        read_ids = list(reader.read_ids)
        rng.shuffle(read_ids)
        for read_id in read_ids:
            if len(signals) >= n_reads:
                break
            try:
                rec  = next(reader.reads([read_id]))
                proc = _preprocess_read(rec.signal_pa)
                if proc is not None:
                    signals.append(proc)
                else:
                    n_rejected += 1
            except Exception:
                n_rejected += 1

    if not signals:
        raise RuntimeError(f"No valid reads from {pod5_path}")

    X = np.stack(signals[:n_reads])
    y = np.full(len(X), label, dtype=np.int32)
    return X, y, n_rejected


def load_dataset(human_pod5_dir: str, pathogen_pod5_dir: str,
                 n_human: int = 50000, n_pathogen: int = 46000,
                 seed: int = 42):
    """
    Load balanced human + pathogen dataset.

    Pathogen reads sampled with 1:2:2:2 ratio:
      S. aureus : E. coli : P. aeruginosa : K. pneumoniae
    """
    import pod5 as _pod5

    # Human
    human_files = sorted(glob.glob(os.path.join(human_pod5_dir, "*.pod5")))
    if not human_files:
        raise FileNotFoundError(f"No POD5 files in: {human_pod5_dir}")

    print(f"\nLoading human reads from {len(human_files)} POD5 files...")
    rng = random.Random(seed)
    rng.shuffle(human_files)
    human_signals, human_rejected = [], 0

    for fpath in human_files:
        if len(human_signals) >= n_human:
            break
        try:
            with _pod5.Reader(fpath) as reader:
                read_ids = list(reader.read_ids)
                rng.shuffle(read_ids)
                for rid in read_ids:
                    if len(human_signals) >= n_human:
                        break
                    try:
                        rec  = next(reader.reads([rid]))
                        proc = _preprocess_read(rec.signal_pa)
                        if proc is not None:
                            human_signals.append(proc)
                        else:
                            human_rejected += 1
                    except Exception:
                        human_rejected += 1
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")

    X_h = np.stack(human_signals[:n_human])
    y_h = np.zeros(len(X_h), dtype=np.int32)
    print(f"  Loaded: {len(X_h):,}  Rejected: {human_rejected:,}")

    # Pathogen
    print(f"\nLoading pathogen reads (1:1:1:1 equal species ratio)...")
    pathogen_arrays, pathogen_rejected, species_counts = [], 0, {}

    for name, spec in SPECIES.items():
        quota = int(np.round(n_pathogen * spec["weight"] / TOTAL_WEIGHT))
        fpath = os.path.join(pathogen_pod5_dir, spec["file"])
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Species POD5 not found: {fpath}")
        X_sp, _, rej = load_pod5_reads(fpath, quota, label=1, seed=seed)
        pathogen_arrays.append(X_sp)
        pathogen_rejected  += rej
        species_counts[name] = len(X_sp)
        print(f"  {name:<14} quota={quota:,}  loaded={len(X_sp):,}  rejected={rej:,}")

    X_p = np.concatenate(pathogen_arrays, axis=0)
    y_p = np.ones(len(X_p), dtype=np.int32)
    print(f"  Total pathogen: {len(X_p):,}  Rejected: {pathogen_rejected:,}")

    # ── Combine & shuffle ─────────────────────────────────────────────────────
    X   = np.concatenate([X_h, X_p], axis=0)
    y   = np.concatenate([y_h, y_p], axis=0)
    idx = np.random.RandomState(seed).permutation(len(y))
    X, y = X[idx], y[idx]

    metadata = {
        "n_human":             int(len(X_h)),
        "n_pathogen":          int(len(X_p)),
        "n_human_rejected":    int(human_rejected),
        "n_pathogen_rejected": int(pathogen_rejected),
        "species_counts":      species_counts,
        "species_ratio":       "1:1:1:1 (equal: S.aureus:E.coli:P.aeruginosa:K.pneumoniae)",
        "window_size":         WINDOW_SIZE,
        "adapter_skip":        ADAPTER_SKIP,
        "human_source":        "GIAB HG002/HG003/HG004 R10.4.1 (ONT open data)",
        "pathogen_source":     "S.aureus/K.pneumoniae/P.aeruginosa/E.coli R10.4.1 (ENA PRJEB51164)",
    }

    print(f"\nDataset ready: {len(X):,} reads ({len(X_h):,} human + {len(X_p):,} pathogen)")
    return X, y, metadata


def get_default_paths():
    home = Path.home()
    return (str(home / "Downloads" / "nanopore" / "human"),
            str(home / "Downloads" / "nanopore" / "pathogen"))
