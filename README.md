# FreeSolo

**Real-time basecall-free pathogen enrichment from raw nanopore signal**

FreeSolo is a CPU-only binary classifier that distinguishes human from microbial DNA directly from raw picoamp nanopore electrical signals, without basecalling. It is designed for deployment with the Oxford Nanopore Technologies [ReadUntil API](https://github.com/nanoporetech/read_until_api), enabling real-time selective sequencing at 99% human background with **0.9950 AUC-ROC** and **21.9x pathogen enrichment**.

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.9950 |
| Accuracy | 0.9671 |
| F1 Score | 0.9669 |
| Mean latency | 4.87 ms |
| Enrichment (99% human) | **21.9x** |

Evaluated on 15,000 held-out real R10.4.1 reads. No GPU required.

---

## Architecture

FreeSolo uses a bank of 15 biologically motivated 1D convolutional filters (edge detectors, Laplacian-of-Gaussian blob detectors, sinusoidal frequency filters, and moving-average baseline filters) to extract a 284-dimensional feature vector from each 4,000-sample signal window. A shallow MLP (256→128→64→2) classifies each vector in under 5 ms on CPU.

The model occupies less than 1 MB of RAM and is architecturally compatible with the MinKNOW ReadUntil API.

---

## Data

### Human reads

Download from the ONT open data repository (GIAB 2023):

```bash
aws s3 cp s3://ont-open-data/giab_2023.05/ ./data/human/ \
    --recursive --no-sign-request \
    --exclude "*" --include "*.pod5"
```

We used samples **HG002**, **HG003**, **HG004** across four PromethION flowcells (runs: 20230424, 20230428, 20230504, 20230503), R10.4.1, LSK114, 5 kHz.

### Pathogen reads

Download from ENA accession [PRJEB51164](https://www.ebi.ac.uk/ena/browser/view/PRJEB51164) (Sanderson et al., Aalborg University). This dataset contains four clinical bacterial species sequenced on a GridION with R10.4.1 flow cells and LSK114 library preparation at 5 kHz — matching the human data exactly.

```bash
# Download FAST5 archives (one per species)
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

```bash
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

```bash
git clone https://github.com/estubin3/FreeSolo.git
cd FreeSolo
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, no GPU needed.

---

## Running the Pipeline

```bash
python run_pipeline.py
```

With custom paths:

```bash
python run_pipeline.py \
    --human-dir /path/to/human/pod5 \
    --pathogen-dir /path/to/pathogen/pod5 \
    --n-human 50000 \
    --n-pathogen 50000
```

Skip the ReadUntil simulation (faster for testing):

```bash
python run_pipeline.py --skip-sim
```

### Pipeline steps

| Step | Description |
|---|---|
| 1 | Load 50k human + 50k pathogen reads from POD5 (12,500/species) |
| 2 | Stratified 70/15/15 train/val/test split |
| 3 | Feature extraction + MLP training |
| 4 | Evaluation on held-out test set |
| 5 | 10-fold cross-validation on 10k subsample |
| 6 | ReadUntil simulation on real held-out signals |

Outputs are written to `results/` and `models/`.

---

## Preprocessing

Each read is preprocessed identically to SquiggleNet [Bao et al., 2021]:

1. **Adapter skip**: discard first 1,500 samples
2. **Quality filter**: discard reads with mean outside [50, 200] pA or std < 1 pA
3. **MAD normalisation**: replace modified z-score outliers (>3.5) with local mean, then standardise to zero mean, unit variance
4. **Window**: extract 4,000-sample (1 second) window

Reads are drawn until the post-QC quota is met per source — rejected reads are replaced on-the-fly, so all reported counts reflect filtered signal.

---

## Project Structure

```
FreeSolo/
├── run_pipeline.py        # Master pipeline
├── classifier.py          # 1D-CNN feature extractor + MLP classifier
├── real_data_loader.py    # POD5 data loader with post-QC quota sampling
├── readuntil_sim.py       # ReadUntil hardware simulator
├── requirements.txt
└── results/               # Output directory (created on first run)
```

---

## Citation

If you use FreeSolo, please cite:

> Stubin, E. (2026). FreeSolo: real-time basecall-free pathogen enrichment from raw nanopore signal.

---

## Acknowledgements

Human sequencing data: GIAB open dataset, Oxford Nanopore Technologies open data repository (registry.opendata.aws/ont-open-data, CC BY-NC 4.0).

Pathogen sequencing data: Sanderson et al., Aalborg University, ENA accession PRJEB51164.

## License

MIT
