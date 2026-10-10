import os, sys, json, subprocess, tempfile
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
import real_data_loader as L
from readuntil_sim import ReadUntilSimulator
import synthetic_data

def _mad_loop(signal):
    """Original per-sample implementation (pre-revision), for equivalence."""
    median = np.median(signal); mad = np.median(np.abs(signal - median)) or 1e-6
    mad = max(mad, 1e-6)
    z = 0.6745 * (signal - median) / mad; m = np.abs(z) > L.MAD_THRESHOLD
    clean = signal.copy()
    for idx in np.where(m)[0]:
        lo, hi = max(0, idx - 2), min(len(signal), idx + 3)
        nb = signal[lo:hi][~m[lo:hi]]
        clean[idx] = np.mean(nb) if len(nb) else median
    std = np.std(clean)
    return (clean - np.mean(clean)) / (std if std > 1e-6 else 1e-6)

def test_mad_vectorised_matches_original():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.normal(100, 5, 4000)
        s[rng.choice(4000, 40)] += rng.normal(0, 200, 40)    # spikes incl. adjacent
        s[100:104] = 900
        assert np.allclose(L._mad_normalise(s), _mad_loop(s), atol=1e-9)

@pytest.fixture(scope="module")
def data():
    d = tempfile.mkdtemp()
    return synthetic_data.make(d, n=40)

def test_loader_provenance_and_quotas(data):
    X, y, meta = L.load_dataset(f"{data}/human", f"{data}/pathogen", n_human=80, n_pathogen=80)
    reads = meta["reads"]
    assert X.shape == (160, 4000) and len(reads) == 160
    hum = [r for r, lab in zip(reads, y) if lab == 0]
    files = {r["file"] for r in hum}
    assert len(files) == 4                                   # every file contributes
    assert {r["kit"] for r, lab in zip(reads, y) if lab == 1} == {"sqk-rbk114-96"}
    assert len({r["read_id"] for r in reads}) == 160         # no duplicates

def test_multi_offset_same_reads(data):
    w, m, _ = L.load_human_reads(f"{data}/human", 20, 0, offsets=(0, 1500, 9500), min_raw_len=13500)
    assert all(w[o].shape == (20, 4000) for o in (0, 1500, 9500))

def test_simulator_control_and_sanity():
    rng = np.random.default_rng(0)
    lh = rng.integers(20000, 400000, 2000); lp = rng.integers(20000, 400000, 2000)
    sim = ReadUntilSimulator(n_pores=128, seed=1)
    perfect = sim.run_simulation_from_probs(np.zeros(500), np.ones(500), lh, lp, 5500, 5, duration_seconds=1200, verbose=False)
    useless = sim.run_simulation_from_probs(np.full(500, .6), np.full(500, .6), lh, lp, 5500, 5, duration_seconds=1200, verbose=False)
    keepall = useless["enrichment"]["relative_enrichment_bases"]
    assert abs(keepall - 1) < 1e-9                           # never ejects -> identical to control
    assert perfect["enrichment"]["relative_enrichment_bases"] > 5
    assert perfect["enrichment"]["absolute_enrichment_bases"] > 1
    assert perfect["bases"]["adaptive_total_bases"] <= perfect["bases"]["control_total_bases"]

def test_diagnostics_smoke(data, tmp_path):
    out = tmp_path / "diag"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "diagnostics.py"), "--data", data,
                        "--out", str(out), "--n-human", "80", "--n-per-species", "30",
                        "--sim-duration", "300"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    res = json.load(open(out / "diagnostics.json"))
    for k in ["A_offset_sweep", "B_batch_probes", "C_positive_control", "E_multi_chunk", "F_latency_ms", "G_simulation", "H_main"]:
        assert k in res
    # synthetic pathogen reads have a shifted first 1,500 samples: offset 0 should be easiest
    assert res["A_offset_sweep"]["0"]["auc"] >= res["A_offset_sweep"]["9500"]["auc"] - 0.05

def test_pipeline_smoke(data, tmp_path):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--human-dir", f"{data}/human",
                        "--pathogen-dir", f"{data}/pathogen", "--n-human", "120", "--n-pathogen", "120",
                        "--n-pathogen-per-species", "30", "--sim-duration", "120"],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

def test_diagnostics_resume(data, tmp_path):
    """With a tiny budget the run stops (exit 3) and later calls finish it."""
    out = tmp_path / "diag_resume"
    cmd = [sys.executable, os.path.join(ROOT, "diagnostics.py"), "--data", data, "--out", str(out),
           "--n-human", "80", "--n-per-species", "30", "--sim-duration", "300", "--budget", "4", "--n-main-human", "80", "--n-main-per-species", "30", "--n-length-survey", "200"]
    codes = []
    for _ in range(60):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        codes.append(r.returncode)
        if r.returncode != 3:
            break
    assert codes[0] == 3 and codes[-1] == 0, (codes, r.stdout[-1500:], r.stderr[-1500:])
    assert "G_simulation" in json.load(open(out / "diagnostics.json"))

def test_basecall_first_paf_parsing(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "baselines"))
    import basecall_first as B
    paf = tmp_path / "x.paf"
    rows = [
        "r1\t1000\t0\t900\t+\tchr1\t1\t0\t900\t850\t900\t60\ttp:A:P",
        "r2\t1000\t0\t900\t+\tecoli_1\t1\t0\t900\t880\t900\t60\ttp:A:P",
        "r2\t1000\t0\t900\t+\tchr2\t1\t0\t900\t300\t900\t5\ttp:A:P",
        "r3\t1000\t0\t900\t+\tsaureus_1\t1\t0\t900\t880\t900\t60\ttp:A:S",
    ]
    paf.write_text("\n".join(rows) + "\n")
    calls = B.classify_paf(str(paf), {"ecoli_1", "saureus_1"})
    assert calls == {"r1": 0, "r2": 1}
    m = B.metrics({"r1": 0, "r2": 1, "r3": 1, "r4": 0}, calls)
    assert m["accuracy"] == 0.75 and m["pathogen_recall"] == 0.5 and m["fraction_mapped"] == 0.5
