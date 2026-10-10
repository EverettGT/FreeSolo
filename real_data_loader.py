"""
Real Data Loader for FreeSolo

Loads raw nanopore signals from POD5 files with controlled per-file and
per-species sampling, and records provenance for every read.

Data actually used (verified from POD5 / FAST5 run metadata, Oct 2026):

  Human (ONT open data, giab_2023.05, HG002/HG003/HG004)
    FLO-PRO114M (R10.4.1), SQK-LSK114 (ligation, no barcodes), 5 kHz
    PromethION 48 (SEBP48) and PromethION 24 (PC24B318)
    3 flow cells / 4 runs:
      PAO89685  run 2264ba8c (2023-04-24)  HG002
      PAO89685  run c9d0d53f (2023-04-28)  HG002  (same flow cell, second run)
      PAO98397  run 289a47d8 (2023-05-04)  HG003
      PAO98633  run 064894cd (2023-05-03)  HG004

  Pathogen (ENA PRJEB51164, runs ERR13848442-ERR13848445)
    FLO-MIN114 (R10.4.1), SQK-RBK114-96 (RAPID BARCODING), 5 kHz
    GridION (GXB04378), ONE flow cell (FAW09960), ONE run (2023-10-05)
      saureus.pod5     - Staphylococcus aureus    (barcode10)
      ecoli.pod5       - Escherichia coli         (barcode09)
      paeruginosa.pod5 - Pseudomonas aeruginosa   (barcode11)
      kpneumoniae.pod5 - Klebsiella pneumoniae    (barcode12)

  Consequence: the class label is confounded with instrument, library kit
  (ligation vs rapid barcoding), flow cell and run. See diagnostics.py.

Preprocessing:
  1. Offset       : discard the first `offset` samples (default 1,500)
  2. Quality filter: window mean outside [50,200] pA or std < 1 pA -> discard
  3. MAD normalise : replace samples with |modified z| > 3.5 by the mean of
                     non-outlier neighbours within +/-2 samples, then
                     zero-mean unit-variance
  4. Window        : `window` samples starting at `offset` (default 4,000)
"""

import numpy as np
import os
import glob
import random
from pathlib import Path

# Preprocessing constants (defaults; offset/window can be overridden per call)
ADAPTER_SKIP  = 1500
WINDOW_SIZE   = 4000
MEAN_MIN_PA   = 50.0
MEAN_MAX_PA   = 200.0
STD_MIN_PA    = 1.0
MAD_THRESHOLD = 3.5
SAMPLE_RATE   = 5000

# Species config
SPECIES = {
    "saureus":     {"file": "saureus.pod5",     "weight": 1},
    "ecoli":       {"file": "ecoli.pod5",        "weight": 1},
    "paeruginosa": {"file": "paeruginosa.pod5",  "weight": 1},
    "kpneumoniae": {"file": "kpneumoniae.pod5",  "weight": 1},
}
TOTAL_WEIGHT = sum(s["weight"] for s in SPECIES.values())  # 4

_KERNEL = np.ones(5)


def _mad_normalise(signal: np.ndarray) -> np.ndarray:
    """MAD outlier replacement + z-normalisation.

    Vectorised version of the original per-sample loop; numerically
    identical (each outlier is replaced by the mean of the non-outlier
    samples in [idx-2, idx+2], or the median if there are none).
    """
    signal = np.asarray(signal, dtype=np.float64)
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    if mad < 1e-6:
        mad = 1e-6
    modified_z = 0.6745 * (signal - median) / mad
    outlier_mask = np.abs(modified_z) > MAD_THRESHOLD
    if np.any(outlier_mask):
        valid = (~outlier_mask).astype(np.float64)
        sums = np.convolve(np.where(outlier_mask, 0.0, signal), _KERNEL, mode="same")
        counts = np.convolve(valid, _KERNEL, mode="same")
        repl = np.where(counts > 0, sums / np.maximum(counts, 1.0), median)
        signal = np.where(outlier_mask, repl, signal)
    std = np.std(signal)
    return (signal - np.mean(signal)) / (std if std > 1e-6 else 1e-6)


def _quality_ok(raw: np.ndarray) -> bool:
    m, s = np.mean(raw), np.std(raw)
    return MEAN_MIN_PA <= m <= MEAN_MAX_PA and s >= STD_MIN_PA


def _preprocess_read(raw: np.ndarray, offset: int = ADAPTER_SKIP,
                     window: int = WINDOW_SIZE):
    if len(raw) < offset + window:
        return None
    win = np.asarray(raw[offset: offset + window], dtype=np.float32)
    if not _quality_ok(win):
        return None
    return _mad_normalise(win).astype(np.float32)


def _record_meta(rec, path, source):
    ri = rec.run_info
    return {
        "read_id":     str(rec.read_id),
        "source":      source,
        "file":        os.path.basename(path),
        "flow_cell":   ri.flow_cell_id,
        "run_id":      ri.acquisition_id[:8],
        "device":      ri.system_name,
        "kit":         ri.sequencing_kit,
        "raw_len":     int(rec.num_samples),
    }


def sample_pod5(pod5_path, n_reads, rng, source, offsets=(ADAPTER_SKIP,),
                window=WINDOW_SIZE, min_raw_len=0, batch=256):
    """Draw up to n_reads reads from one POD5 file.

    A read is accepted only if it passes QC at EVERY offset in `offsets`
    and is at least `min_raw_len` samples long, so that multi-offset
    experiments use exactly the same reads at each offset.

    Returns (dict offset -> list of windows, list of metadata dicts, n_rejected).
    """
    import pod5
    windows = {o: [] for o in offsets}
    meta, n_rejected = [], 0
    with pod5.Reader(pod5_path) as reader:
        read_ids = [str(r) for r in reader.read_ids]
        rng.shuffle(read_ids)
        for start in range(0, len(read_ids), batch):
            if len(meta) >= n_reads:
                break
            chunk = read_ids[start:start + batch]
            for rec in reader.reads(selection=chunk, missing_ok=True):
                if len(meta) >= n_reads:
                    break
                try:
                    if rec.num_samples < min_raw_len:
                        n_rejected += 1
                        continue
                    raw = rec.signal_pa
                    procs = [_preprocess_read(raw, o, window) for o in offsets]
                    if any(p is None for p in procs):
                        n_rejected += 1
                        continue
                    for o, p in zip(offsets, procs):
                        windows[o].append(p)
                    meta.append(_record_meta(rec, pod5_path, source))
                except Exception:
                    n_rejected += 1
    return windows, meta, n_rejected


def load_pod5_reads(pod5_path: str, n_reads: int, label: int, seed: int = 42,
                    offset: int = ADAPTER_SKIP, window: int = WINDOW_SIZE,
                    return_meta: bool = False, source: str = None):
    """Load up to n_reads from a single POD5 file."""
    rng = random.Random(seed)
    src = source or os.path.splitext(os.path.basename(pod5_path))[0]
    wins, meta, n_rejected = sample_pod5(pod5_path, n_reads, rng, src,
                                         offsets=(offset,), window=window)
    if not meta:
        raise RuntimeError(f"No valid reads from {pod5_path}")
    X = np.stack(wins[offset])
    y = np.full(len(X), label, dtype=np.int32)
    if return_meta:
        return X, y, n_rejected, meta
    return X, y, n_rejected


def load_human_reads(human_pod5_dir, n_human, seed=42, offsets=(ADAPTER_SKIP,),
                     window=WINDOW_SIZE, min_raw_len=0):
    """Equal per-file quotas across all human POD5 files.

    (The original loader filled the quota file-by-file after shuffling, so
    reads could come from only a subset of files and flow cells.)
    """
    human_files = sorted(glob.glob(os.path.join(human_pod5_dir, "*.pod5")))
    if not human_files:
        raise FileNotFoundError(f"No POD5 files in: {human_pod5_dir}")
    rng = random.Random(seed)
    per_file = int(np.ceil(n_human / len(human_files)))
    wins = {o: [] for o in offsets}
    meta, rejected = [], 0
    for fpath in human_files:
        w, m, r = sample_pod5(fpath, per_file, rng, "human", offsets, window, min_raw_len)
        for o in offsets:
            wins[o].extend(w[o])
        meta.extend(m)
        rejected += r
    # trim to exactly n_human, keeping per-file balance as far as possible
    keep = sorted(rng.sample(range(len(meta)), min(n_human, len(meta))))
    wins = {o: np.stack([wins[o][i] for i in keep]) for o in offsets}
    meta = [meta[i] for i in keep]
    return wins, meta, rejected


def load_pathogen_reads(pathogen_pod5_dir, n_per_species, seed=42,
                        offsets=(ADAPTER_SKIP,), window=WINDOW_SIZE, min_raw_len=0):
    wins = {o: [] for o in offsets}
    meta, rejected, counts = [], 0, {}
    for name, spec in SPECIES.items():
        fpath = os.path.join(pathogen_pod5_dir, spec["file"])
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Species POD5 not found: {fpath}")
        rng = random.Random(seed)
        w, m, r = sample_pod5(fpath, n_per_species, rng, name, offsets, window, min_raw_len)
        for o in offsets:
            wins[o].extend(w[o])
        meta.extend(m)
        rejected += r
        counts[name] = len(m)
    wins = {o: np.stack(v) for o, v in wins.items()}
    return wins, meta, rejected, counts


def load_dataset(human_pod5_dir: str, pathogen_pod5_dir: str,
                 n_human: int = 50000, n_pathogen: int = 50000,
                 seed: int = 42, offset: int = ADAPTER_SKIP,
                 window: int = WINDOW_SIZE):
    """
    Load balanced human + pathogen dataset.

    Human reads: equal quota per POD5 file. Pathogen reads: 1:1:1:1 species.
    metadata["reads"] holds one provenance dict per row of X (same order).
    """
    print(f"\nLoading human reads (equal per-file quotas)...")
    hw, h_meta, h_rej = load_human_reads(human_pod5_dir, n_human, seed, (offset,), window)
    X_h = hw[offset]
    print(f"  Loaded: {len(X_h):,}  Rejected: {h_rej:,}")

    print(f"\nLoading pathogen reads (1:1:1:1 equal species ratio)...")
    quota = int(np.round(n_pathogen / TOTAL_WEIGHT))
    pw, p_meta, p_rej, counts = load_pathogen_reads(pathogen_pod5_dir, quota, seed, (offset,), window)
    X_p = pw[offset]
    for k, v in counts.items():
        print(f"  {k:<14} loaded={v:,}")

    X = np.concatenate([X_h, X_p], axis=0)
    y = np.concatenate([np.zeros(len(X_h), np.int32), np.ones(len(X_p), np.int32)])
    reads = h_meta + p_meta
    idx = np.random.RandomState(seed).permutation(len(y))
    X, y = X[idx], y[idx]
    reads = [reads[i] for i in idx]

    metadata = {
        "n_human":             int(len(X_h)),
        "n_pathogen":          int(len(X_p)),
        "n_human_rejected":    int(h_rej),
        "n_pathogen_rejected": int(p_rej),
        "species_counts":      counts,
        "species_ratio":       "1:1:1:1 (equal: S.aureus:E.coli:P.aeruginosa:K.pneumoniae)",
        "window_size":         window,
        "offset":              offset,
        "human_source":        "GIAB HG002/HG003/HG004, FLO-PRO114M R10.4.1, SQK-LSK114, PromethION, 5 kHz (ONT open data)",
        "pathogen_source":     "S.aureus/K.pneumoniae/P.aeruginosa/E.coli, FLO-MIN114 R10.4.1, SQK-RBK114-96, GridION, 5 kHz, one flow cell (ENA PRJEB51164)",
        "reads":               reads,
    }
    print(f"\nDataset ready: {len(X):,} reads ({len(X_h):,} human + {len(X_p):,} pathogen)")
    return X, y, metadata


def save_read_table(reads, path):
    """Write per-read provenance (read ID, flow cell, run, ...) as CSV."""
    import csv
    if not reads:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reads[0].keys()))
        w.writeheader()
        w.writerows(reads)


def get_default_paths():
    """Default data location; override with FREESOLO_DATA or CLI flags."""
    base = os.environ.get("FREESOLO_DATA")
    base = Path(base) if base else Path.home() / "Downloads" / "nanopore"
    return str(base / "human"), str(base / "pathogen")
