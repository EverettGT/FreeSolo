"""
Basecall-first baseline (Reviewer 2, comment 3): Dorado + minimap2 on the
same reads FreeSolo is evaluated on.

Each read is basecalled with Dorado, optionally truncated to its first K
bases (to mimic an adaptive-sampling decision made after K bases), aligned
with minimap2 to a combined reference (human + the four bacterial genomes),
and called "pathogen" if its best primary alignment is to a bacterial contig.
Unmapped reads are called "human" (they would be ejected / not kept).

This could not be run in the environment where the revision was prepared
(no access to ONT's download server or NCBI), so it is provided ready to run.

Example (Linux/macOS/WSL):

  python baselines/basecall_first.py \
      --reads results/revision/reads_main.csv \
      --data ~/Downloads/nanopore \
      --dorado /opt/dorado/bin/dorado --model hac \
      --minimap2 minimap2 \
      --human-ref GRCh38.fa --pathogen-ref bacteria.fa \
      --out results/revision/basecall_first --device cpu --max-reads 2000

  bacteria.fa = the four reference genomes concatenated (any contig names).
  Use --device cuda:0 on a GPU machine; report both CPU and GPU timings.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np


def fasta_contigs(path):
    names = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                names.append(line[1:].split()[0])
    return names


def subset_pod5(reads, data_dir, out_pod5):
    """Write the selected read IDs to one POD5 file."""
    import pod5
    by_file = defaultdict(list)
    for r in reads:
        sub = "human" if r["source"] == "human" else "pathogen"
        by_file[os.path.join(data_dir, sub, r["file"])].append(r["read_id"])
    n = 0
    with pod5.Writer(out_pod5) as w:
        for path, ids in by_file.items():
            with pod5.Reader(path) as rd:
                for rec in rd.reads(selection=ids, missing_ok=True):
                    w.add_read(rec.to_read())
                    n += 1
    return n


def read_fastq(path):
    seqs = {}
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline().strip()
            f.readline(); f.readline()
            seqs[h[1:].split()[0]] = s
    return seqs


def write_fasta(seqs, path, k=None):
    with open(path, "w") as f:
        for rid, s in seqs.items():
            s = s[:k] if k else s
            if s:
                f.write(f">{rid}\n{s}\n")


def classify_paf(paf_path, pathogen_contigs):
    """Best primary alignment per read (highest MAPQ, then matches)."""
    best = {}
    with open(paf_path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            tags = c[12:]
            if any(t == "tp:A:S" for t in tags):
                continue
            rid, target, matches, mapq = c[0], c[5], int(c[9]), int(c[11])
            key = (mapq, matches)
            if rid not in best or key > best[rid][0]:
                best[rid] = (key, target)
    return {rid: (1 if t in pathogen_contigs else 0) for rid, (_, t) in best.items()}


def metrics(labels, calls):
    y = np.array([labels[r] for r in labels])
    p = np.array([calls.get(r, 0) for r in labels])        # unmapped -> human
    mapped = np.array([r in calls for r in labels])
    return {
        "n": int(len(y)),
        "accuracy": float((p == y).mean()),
        "pathogen_recall": float(p[y == 1].mean()) if (y == 1).any() else None,
        "human_recall": float((1 - p[y == 0]).mean()) if (y == 0).any() else None,
        "fraction_mapped": float(mapped.mean()),
        "fraction_mapped_human": float(mapped[y == 0].mean()) if (y == 0).any() else None,
        "fraction_mapped_pathogen": float(mapped[y == 1].mean()) if (y == 1).any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True, help="reads_main.csv from diagnostics.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dorado", required=True)
    ap.add_argument("--model", default="hac", help="fast / hac / sup or a model path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--minimap2", default="minimap2")
    ap.add_argument("--human-ref", required=True)
    ap.add_argument("--pathogen-ref", required=True)
    ap.add_argument("--out", default="results/revision/basecall_first")
    ap.add_argument("--max-reads", type=int, default=2000, help="per class")
    ap.add_argument("--prefix-bases", default="450,1000,0",
                    help="comma list; 0 = full read")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.reads)))
    rng = np.random.RandomState(a.seed)
    sel = []
    for cls in ("human", "pathogen"):
        pool = [r for r in rows if (r["source"] == "human") == (cls == "human")]
        idx = rng.permutation(len(pool))[: a.max_reads]
        sel += [pool[i] for i in idx]
    labels = {r["read_id"]: 0 if r["source"] == "human" else 1 for r in sel}

    pod5_path = os.path.join(a.out, "subset.pod5")
    n = subset_pod5(sel, a.data, pod5_path)
    print(f"subset: {n} reads")

    fq = os.path.join(a.out, "calls.fastq")
    t0 = time.time()
    with open(fq, "w") as f:
        subprocess.run([a.dorado, "basecaller", a.model, pod5_path, "--device", a.device,
                        "--emit-fastq"], stdout=f, check=True)
    t_bc = time.time() - t0
    seqs = read_fastq(fq)
    total_bases = sum(len(s) for s in seqs.values())

    ref = os.path.join(a.out, "combined_ref.fa")
    with open(ref, "w") as out:
        for p in (a.human_ref, a.pathogen_ref):
            with open(p) as f:
                for line in f:
                    out.write(line)
    pathogen_contigs = set(fasta_contigs(a.pathogen_ref))
    mmi = os.path.join(a.out, "combined_ref.mmi")
    if not os.path.exists(mmi):
        subprocess.run([a.minimap2, "-x", "map-ont", "-t", str(a.threads), "-d", mmi, ref], check=True)

    results = {"model": a.model, "device": a.device, "n_basecalled": len(seqs),
               "basecall_seconds": t_bc, "bases": total_bases,
               "basecall_bases_per_s": total_bases / t_bc if t_bc else None,
               "basecall_reads_per_s": len(seqs) / t_bc if t_bc else None, "by_prefix": {}}
    for k in [int(x) for x in a.prefix_bases.split(",")]:
        fa = os.path.join(a.out, f"prefix_{k}.fa")
        write_fasta(seqs, fa, k or None)
        paf = os.path.join(a.out, f"prefix_{k}.paf")
        t0 = time.time()
        with open(paf, "w") as f:
            subprocess.run([a.minimap2, "-x", "map-ont", "-t", str(a.threads), "--secondary=no",
                            mmi, fa], stdout=f, check=True)
        res = metrics(labels, classify_paf(paf, pathogen_contigs))
        res["map_seconds"] = time.time() - t0
        results["by_prefix"]["full" if k == 0 else str(k)] = res
        print(f"prefix {k or 'full'}: {res}")
    json.dump(results, open(os.path.join(a.out, "basecall_first.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
