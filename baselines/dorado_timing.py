"""
CPU timing: Dorado basecalling vs FreeSolo on the same read prefixes.

FreeSolo decides after 5,500 samples (1,500-sample skip + 4,000-sample
window, 1.1 s at 5 kHz). This script takes N reads from the main evaluation
(results/revision/reads_main.csv), cuts each to its first 5,500 samples,
writes them to one POD5 file, and times on the same machine:

  * Dorado basecalling of those prefixes on the CPU (after a warm-up run that
    also downloads the model), pinned to --cores CPU cores
  * FreeSolo preprocessing + feature extraction + classification of the same
    prefixes, single-threaded

Usage (Linux, WSL or Windows; on Windows --cores is ignored and all cores are used):
  python baselines/dorado_timing.py --data "/mnt/c/Users/<you>/Downloads/FreeSolo Bioinformatics Project/nanopore" \
      --dorado ~/dorado/bin/dorado --model fast --cores 2 --n 300

Writes results/revision/dorado_timing.json and prints a summary.
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREFIX = 5500


def write_prefixes(rows, data_dir, out_pod5):
    import pod5
    by_file = defaultdict(list)
    for r in rows:
        sub = "human" if r["source"] == "human" else "pathogen"
        by_file[os.path.join(data_dir, sub, r["file"])].append(r["read_id"])
    raw = []
    if os.path.exists(out_pod5):
        os.remove(out_pod5)
    with pod5.Writer(out_pod5) as w:
        for path, ids in by_file.items():
            with pod5.Reader(path) as rd:
                for rec in rd.reads(selection=ids, missing_ok=True):
                    read = rec.to_read()
                    read.signal = read.signal[:PREFIX].copy()
                    w.add_read(read)
                    raw.append(rec.signal_pa[:PREFIX].copy())
    return raw


def run_dorado(dorado, model, pod5_path, cores, out_fastq):
    cmd = [dorado, "basecaller", model, pod5_path, "--device", "cpu", "--emit-fastq"]
    if cores and shutil.which("taskset"):
        cmd = ["taskset", "-c", ",".join(str(i) for i in range(cores))] + cmd
    t0 = time.perf_counter()
    log = out_fastq.replace(".fastq", ".log")
    with open(out_fastq, "w") as f, open(log, "w") as err:
        r = subprocess.run(cmd, stdout=f, stderr=err)
    if r.returncode != 0:
        sys.exit(f"Dorado failed (exit {r.returncode}); see {log}")
    return time.perf_counter() - t0


def count_fastq(path):
    n, bases = 0, 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                n += 1
                bases += len(line.strip())
    return n, bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with human/ and pathogen/ POD5 files")
    ap.add_argument("--reads", default="results/revision/reads_main.csv")
    ap.add_argument("--dorado", required=True)
    ap.add_argument("--model", default="fast", help="fast / hac / sup")
    ap.add_argument("--cores", type=int, default=2, help="pin Dorado to this many cores (0 = all)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/revision/dorado_timing")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.reads)))
    rng = np.random.RandomState(a.seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[: a.n]]
    pod5_path = os.path.join(a.out, "prefixes.pod5")
    raw = write_prefixes(rows, a.data, pod5_path)
    print(f"wrote {len(raw)} read prefixes of {PREFIX} samples")

    # warm-up (downloads the model, loads libraries) on a small file
    warm = os.path.join(a.out, "warmup.pod5")
    write_prefixes(rows[:10], a.data, warm)
    print("Dorado warm-up (first run downloads the model)...")
    run_dorado(a.dorado, a.model, warm, a.cores, os.path.join(a.out, "warmup.fastq"))

    print(f"Timing Dorado ({a.model}, CPU, {a.cores or 'all'} cores) on {len(raw)} prefixes...")
    t_dorado = run_dorado(a.dorado, a.model, pod5_path, a.cores, os.path.join(a.out, "prefixes.fastq"))
    n_called, bases = count_fastq(os.path.join(a.out, "prefixes.fastq"))

    print("Timing FreeSolo (single thread) on the same prefixes...")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from real_data_loader import _preprocess_read
    from classifier import Conv1DFeatureExtractor
    from sklearn.linear_model import LogisticRegression
    fe = Conv1DFeatureExtractor(); fe.verbose = False
    clf = LogisticRegression().fit(np.random.randn(50, 280), np.r_[np.zeros(25), np.ones(25)])
    lat = []
    for x in raw:
        t0 = time.perf_counter()
        w = _preprocess_read(np.asarray(x))
        if w is not None:
            clf.predict_proba(fe.extract_features(w.reshape(1, -1)))
            lat.append((time.perf_counter() - t0) * 1000)

    res = {
        "n_prefixes": len(raw), "prefix_samples": PREFIX, "prefix_seconds_of_signal": PREFIX / 5000,
        "dorado": {"model": a.model, "cores": a.cores or os.cpu_count(), "wall_seconds": t_dorado,
                   "reads_basecalled": n_called, "bases": bases,
                   "seconds_per_read": t_dorado / max(1, len(raw)),
                   "samples_per_second": len(raw) * PREFIX / t_dorado,
                   "realtime_channels_supported": (len(raw) * PREFIX / t_dorado) / 5000},
        "freesolo": {"threads": 1, "n_decisions": len(lat), "ms_per_read_mean": float(np.mean(lat)),
                     "ms_per_read_p95": float(np.percentile(lat, 95)),
                     "realtime_channels_supported": 1000 * (PREFIX / 5000) / float(np.mean(lat))},
        "host": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
    }
    json.dump(res, open(os.path.join(a.out, "dorado_timing.json"), "w"), indent=2)
    d, f = res["dorado"], res["freesolo"]
    print("\n==== RESULT (paste this back) ====")
    print(json.dumps(res, indent=2))
    print(f"\nDorado {a.model}: {d['seconds_per_read']*1000:.0f} ms per 1.1 s prefix on {d['cores']} cores "
          f"(keeps up with ~{d['realtime_channels_supported']:.1f} channels)")
    print(f"FreeSolo: {f['ms_per_read_mean']:.1f} ms per prefix on 1 thread "
          f"(keeps up with ~{f['realtime_channels_supported']:.0f} channels)")


if __name__ == "__main__":
    main()
