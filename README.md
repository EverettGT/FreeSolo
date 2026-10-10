# FreeSolo

**Real-time basecall-free pathogen enrichment from raw nanopore signal**

FreeSolo is a CPU-only binary classifier that distinguishes human from microbial DNA directly from raw picoamp nanopore electrical signals, without basecalling. It is designed for deployment with the Oxford Nanopore Technologies [ReadUntil API](https://github.com/nanoporetech/read_until_api), enabling real-time selective sequencing at 99% human background.

---

## Results

Results from the October 2026 revision (`diagnostics.py`, results in `results/revision/diagnostics.json`). Every test set uses human reads from a PromethION flow cell that was never seen in training.

| Metric | Value |
| --- | --- |
| AUC-ROC, each human flow cell held out in turn (mean ± sd, 3 folds) | 0.990 ± 0.002 |
| Accuracy, same folds | 0.951 ± 0.008 |
| AUC-ROC with 1 / 2 / 3 consecutive 4,000-sample chunks (reads of at least 13,500 samples, held-out flow cell PAO98633) | 0.974 / 0.985 / 0.990 |
| Time to decision (1 chunk) | 1.1 s of signal + 6.3 ms compute (preprocessing included, 2 CPU threads) |
| Relative enrichment in sequenced bases at 99% human (simulated) | 9.7x to 13.9x |
| Pathogen yield gain over a no-ejection control (simulated) | 3.9x to 11.6x, depending on capture and unblock times |

Main-evaluation reads: 12,000 human (all 20 POD5 files) and 12,000 pathogen (3,000 per species). Simulated values come from an idealized model (no pore wear, no channel noise), so treat them as upper bounds.

The first submission reported AUC 0.995 / accuracy 0.967 on a read-level random split and a 25.1x read-level enrichment from a simulator with a baseline accounting error; those numbers are kept in `results/` for reference only.

### Confound checks

Human reads (PromethION, ligation kit SQK-LSK114) and pathogen reads (GridION, rapid barcoding kit SQK-RBK114-96, one flow cell) differ in instrument, library kit, flow cell and run, so these checks ask how much of the accuracy could come from those differences:

| Check | Result |
| --- | --- |
| Window moved deeper into the read (1,500 → 3,000 → 5,500 → 9,500 samples) | AUC 0.974 → 0.975 → 0.980 → 0.978: no drop, so the classifier is not relying on adapter or barcode signal at the read start. A window at sample 0 (adapter included) gives 0.997. |
| Can the features tell two human runs apart (same flow cell and sample; different flow cells; different PromethION models)? | AUC 0.51 to 0.57: run, flow cell and PromethION model leave little trace in the features. |
| Species vs species on the single pathogen flow cell (platform held constant) | AUC tracks GC difference: S. aureus vs P. aeruginosa 0.97, S. aureus vs E. coli 0.89, E. coli vs K. pneumoniae 0.64, P. aeruginosa vs K. pneumoniae 0.67. |
| Leave-one-species-out with an unseen human flow cell | AUC 0.955 to 0.993; S. aureus pathogen recall 0.70. |

What these checks cannot rule out: an effect of the GridION instrument or the rapid barcoding kit itself, because no human reads from that instrument and kit were available. A matched-platform dataset is needed for that.

## Architecture

FreeSolo uses a bank of 15 biologically motivated 1D convolutional filters (edge detectors, Laplacian-of-Gaussian blob detectors, sinusoidal frequency filters, and moving-average baseline filters) to extract a 280-dimensional feature vector from each 4,000-sample signal window. A shallow MLP (256→128→64→2) classifies each vector in under 10 ms on CPU.

The filters are fixed, hand-designed kernels, not learned weights. The classifier has not yet been run against a live MinKNOW ReadUntil stream.

---

## Data

### Human reads

Download from the ONT open data repository (GIAB 2023):

```
aws s3 cp s3://ont-open-data/giab_2023.05/ ./data/human/ \
    --recursive --no-sign-request \
    --exclude "*" --include "*.pod5"
```

We used samples **HG002**, **HG003**, **HG004**: four runs on three PromethION flow cells (FLO-PRO114M, R10.4.1; PAO89685 twice, PAO98397, PAO98633), SQK-LSK114 ligation kit, 5 kHz, on PromethION 48 and PromethION 24 instruments. These values come from the run metadata in the POD5 files.

### Pathogen reads

Download from ENA accession [PRJEB51164](https://www.ebi.ac.uk/ena/browser/view/PRJEB51164) (Sanderson et al., 2023, University of Oxford). The runs used here (barcodes 09 to 12) are one multiplexed run on a single GridION flow cell (FLO-MIN114, R10.4.1, flow cell FAW09960) with the **Rapid Barcoding Kit SQK-RBK114-96** at 5 kHz (Sanderson et al., 2024, Microbial Genomics 10:001246). The library kit and instrument therefore differ from the human data; see the confound checks above.

```
curl -o data/pathogen/s_aureus.tar.gz \
    "https://ftp.sra.ebi.ac.uk/vol1/run/ERR138/ERR13848443/barcode10.tar.gz"

curl -o data/pathogen/k_pneumoniae.tar.gz \
    "https://ftp.sra.ebi.ac.uk/vol1/run/ERR138/ERR13848442/barcode12.tar.gz"

curl -o data/pathogen/p_aeruginosa.tar.gz \
    "https://ftp.sra.ebi.ac.uk/vol1/run/ERR138/ERR13848444/barcode11.tar.gz"

curl -o data/pathogen/e_coli.tar.gz \
    "https://ftp.sra.ebi.ac.uk/vol1/run/ERR138/ERR13848445/barcode09.tar.gz"
```

Extract and convert each species from FAST5 to POD5:

```
cd data/pathogen

tar -xzf s_aureus.tar.gz
tar -xzf k_pneumoniae.tar.gz
tar -xzf p_aeruginosa.tar.gz
tar -xzf e_coli.tar.gz

pip install pod5

pod5 convert fast5 barcode10/ --output saureus.pod5 --force-overwrite
pod5 convert fast5 barcode12/ --output kpneumoniae.pod5 --force-overwrite
pod5 convert fast5 barcode11/ --output paeruginosa.pod5 --force-overwrite
pod5 convert fast5 barcode09/ --output ecoli.pod5 --force-overwrite
```

### Expected directory layout

```
Downloads/nanopore/
├── human/
│   ├── PAO89685_pass__*.pod5   (20 files, GIAB HG002/3/4)
│   └── ...
└── pathogen/
    ├── saureus.pod5
    ├── kpneumoniae.pod5
    ├── paeruginosa.pod5
    └── ecoli.pod5
```

The pipeline auto-detects these paths from `~/Downloads/nanopore/`. Override with `--human-dir` and `--pathogen-dir`.

---

## Installation

```
git clone https://github.com/EverettGT/FreeSolo.git
cd FreeSolo
pip install -r requirements.txt
```

**Requirements:** Python 3.9 or later, no GPU needed.

---

## Running the Pipeline

```
python run_pipeline.py
```

With custom paths:

```
python run_pipeline.py \
    --human-dir /path/to/human/pod5 \
    --pathogen-dir /path/to/pathogen/pod5 \
    --n-human 50000 \
    --n-pathogen 50000
```

Skip the ReadUntil simulation or holdout experiment if you want faster runs:

```
python run_pipeline.py --skip-sim
python run_pipeline.py --skip-holdout
python run_pipeline.py --skip-sim --skip-holdout
```

### Confound diagnostics (revision)

```
python diagnostics.py --data ~/Downloads/nanopore --out results/revision --budget 160
```

Every step is checkpointed. With `--budget`, the script stops before the time limit and exits with code 3; run the same command again to continue. This is how the revision results were produced on a 2-core laptop VM.

The main pipeline now holds out a human flow cell for testing by default (`--split flowcell`); `--split read` reproduces the original read-level split.

### Basecall-first baseline

`baselines/basecall_first.py` runs Dorado + minimap2 on the same reads (`results/revision/reads_main.csv`) and reports accuracy after the first 450 / 1,000 bases and for full reads, with basecalling throughput. It needs Dorado, minimap2, GRCh38 and the four bacterial reference genomes.

### Tests

```
python -m pytest tests
```

### Pipeline steps

| Step | Description |
| --- | --- |
| 1 | Load 50k human + 50k pathogen reads from POD5 (12,500/species) |
| 2 | Split: human test reads from a held-out flow cell (default) or read-level 70/15/15 (`--split read`) |
| 3 | Feature extraction + MLP training |
| 4 | Evaluation on held-out test set |
| 5 | 10-fold cross-validation on 10k training subsample |
| 6 | ReadUntil simulation: control vs adaptive run, enrichment in bases, empirical read lengths |
| 7 | Leave-one-species-out generalization experiment (4 folds) |

Main results write to `results/`. Holdout fold results write to `results/holdout_species/`.

---

## Preprocessing

Each read is preprocessed identically to SquiggleNet (Bao et al., 2021):

1. **Adapter skip**: discard first 1,500 samples
2. **Quality filter**: discard reads with mean outside [50, 200] pA or std < 1 pA
3. **MAD normalization**: replace modified z-score outliers (>3.5) with local mean, then standardize to zero mean, unit variance
4. **Window**: extract 4,000-sample (0.8 second at 5 kHz) window

Reads are drawn until the post-QC quota is met per source (rejected reads are replaced on the fly), with equal quotas per human POD5 file. Every loaded read is recorded with its read ID, flow cell, run, kit and length (`reads.csv`).

---

## Project Structure

```
FreeSolo/
├── run_pipeline.py
├── classifier.py
├── real_data_loader.py
├── readuntil_sim.py
├── diagnostics.py
├── baselines/basecall_first.py
├── tests/
├── requirements.txt
└── results/
    └── holdout_species/
```

---

## Citation

If you use FreeSolo, please cite:
> Stubin, E. (2026). FreeSolo: real-time basecall-free pathogen enrichment from raw nanopore signal.

---

## Acknowledgements

Human sequencing data: GIAB open dataset, Oxford Nanopore Technologies open data repository (registry.opendata.aws/ont-open-data, CC BY-NC 4.0).

Pathogen sequencing data: Sanderson et al., 2024 (Microbial Genomics 10:001246), University of Oxford, ENA accession PRJEB51164.

## License

MIT
