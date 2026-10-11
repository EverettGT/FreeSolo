# FreeSolo

**Real-time basecall-free pathogen enrichment from raw nanopore signal**

FreeSolo is a CPU-only binary classifier that distinguishes human from microbial DNA directly from raw picoamp nanopore electrical signals, without basecalling. It is designed for deployment with the Oxford Nanopore Technologies [ReadUntil API](https://github.com/nanoporetech/read_until_api), enabling real-time selective sequencing at 99% human background.

---

## Results

Results from the October 2026 revision. The classifier is logistic regression on 280 signal features (`diagnostics.py --clf logreg`, results in `results/revision/logreg/diagnostics.json`). Every test set uses human reads from a PromethION flow cell that was never seen in training.

| Metric | Value |
| --- | --- |
| AUC-ROC, each human flow cell held out in turn (mean ± sd, 3 folds) | 0.989 ± 0.001 |
| Accuracy, same folds | 0.960 ± 0.004 |
| Calibration on the PAO98633 fold (Brier / ECE) | 0.030 / 0.023 |
| AUC-ROC with 1 / 2 / 3 consecutive 4,000-sample chunks (reads of at least 13,500 samples, held-out flow cell PAO98633) | 0.984 / 0.994 / 0.996 |
| Time to decision (1 chunk) | 1.1 s of signal + 3 to 6 ms compute (preprocessing included, 2 CPU threads; the classifier itself is about 0.1 ms) |
| Relative enrichment in sequenced bases at 99% human (simulated) | 12.2x to 17.7x |
| Pathogen yield gain over a no-ejection control (simulated) | 4.1x to 14.1x, depending on capture and unblock times |

Main-evaluation reads: 12,000 human (all 20 POD5 files) and 12,000 pathogen (3,000 per species). Simulated values come from an idealized model (no pore wear, no channel noise), so treat them as upper bounds.

The MLP used in the first submission is still available (`--model mlp` / `--clf mlp`); its revision results are in `results/revision/diagnostics.json`. The first submission reported AUC 0.995 / accuracy 0.967 on a read-level random split and a 25.1x read-level enrichment from a simulator with a baseline accounting error; those numbers are kept in `results/` for reference only.

### What the model needs (ablation)

`ablation.py` and `filter_controls.py`, results in `results/revision/ABLATION.md`.

| Features | Logistic regression AUC | MLP AUC |
| --- | --- | --- |
| Raw 4,000-sample window | 0.505 | 0.539 |
| 70 statistical features | 0.978 | 0.985 |
| 210 designed filter features | 0.892 | 0.892 |
| 210 random-kernel features (5 banks) | 0.902 to 0.948 | |
| Statistical + designed filters (FreeSolo) | 0.989 | 0.990 |
| Statistical + random kernels (5 banks) | 0.987 to 0.990 | |

- Feature extraction is necessary; a linear model is enough.
- The convolutional features add +0.011 AUC over the statistics (all 15 paired test sets).
- Random kernels of the same lengths add as much as the hand-designed filters, so the gain comes from convolution and pooling, not from the specific filter shapes (compare ROCKET, Dempster et al. 2020).

### Confound checks

Human reads (PromethION, ligation kit SQK-LSK114) and pathogen reads (GridION, rapid barcoding kit SQK-RBK114-96, one flow cell) differ in instrument, library kit, flow cell and run, so these checks ask how much of the accuracy could come from those differences:

| Check | Result |
| --- | --- |
| Window moved deeper into the read (1,500 → 3,000 → 5,500 → 9,500 samples) | AUC 0.984 → 0.986 → 0.986 → 0.986: no drop, so the classifier is not relying on adapter or barcode signal at the read start. A window at sample 0 (adapter included) gives 0.992. |
| Can the features tell human reads apart by flow cell or PromethION model? | AUC 0.535 and 0.563: little trace. |
| Can they tell two runs on the same flow cell and sample apart? | AUC 0.706 (0.574 with the MLP): some run-level information, possibly pore wear between runs. |
| Species vs species on the single pathogen flow cell (platform held constant) | AUC tracks GC difference: S. aureus vs P. aeruginosa 0.97, S. aureus vs E. coli 0.89, E. coli vs K. pneumoniae 0.65, P. aeruginosa vs K. pneumoniae 0.68. |
| Leave-one-species-out with an unseen human flow cell | AUC 0.968 to 0.990; S. aureus pathogen recall 0.79. |

What these checks cannot rule out: an effect of the GridION instrument or the rapid barcoding kit itself, because no human reads from that instrument and kit were available. A matched-platform dataset is needed for that.

## Architecture

Each 4,000-sample window is convolved with 15 fixed 1D filters (edge, Laplacian-of-Gaussian, sinusoidal and moving-average kernels); each output is ReLU-activated and pooled into 14 values (210 features). Seventy statistical features (moments, segment means and deviations, difference statistics, spectral band energies, autocorrelations) are added, the 280 features are standardized, and an L2-regularized logistic regression makes the decision.

The filters are fixed kernels, not learned weights, and the ablation shows their exact shapes do not matter. The classifier has not yet been run against a live MinKNOW ReadUntil stream.

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
python diagnostics.py --data ~/Downloads/nanopore --out results/revision/logreg --clf logreg --budget 160
python ablation.py --cache <diagnostics cache dir> --budget 150
python filter_controls.py --cache <diagnostics cache dir> --budget 150
```

Every step is checkpointed. With `--budget`, each script stops before the time limit and exits with code 3; run the same command again to continue. This is how the revision results were produced on a 2-core laptop VM.

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
| 3 | Feature extraction + classifier training (logistic regression by default, `--model mlp` for the MLP) |
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
├── ablation.py
├── filter_controls.py
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
