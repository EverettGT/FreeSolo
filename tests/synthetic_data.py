"""Write small synthetic POD5 files mimicking the real layout (for tests only)."""
import os, uuid, datetime
import numpy as np
import pod5

def _run_info(flow_cell, run, device, kit, product):
    now = datetime.datetime(2023, 5, 1, tzinfo=datetime.timezone.utc)
    return pod5.RunInfo(
        acquisition_id=run + "0" * 32, acquisition_start_time=now, adc_max=2047, adc_min=-2048,
        context_tags={"sample_frequency": "5000"}, experiment_name="", flow_cell_id=flow_cell,
        flow_cell_product_code=product, protocol_name="test", protocol_run_id="x",
        protocol_start_time=now, sample_id="s", sample_rate=5000, sequencing_kit=kit,
        sequencer_position="X1", sequencer_position_type="test", software="test",
        system_name=device, system_type="test", tracking_id={})

def write_file(path, n, rng, run_info, level, jitter, prefix_shift=0.0):
    cal = pod5.Calibration(offset=0.0, scale=0.1)
    with pod5.Writer(path) as w:
        for i in range(n):
            L = int(rng.integers(14000, 40000))
            pa = level + jitter * rng.standard_normal(L) + 5 * np.sin(np.arange(L) / (7 + 3 * jitter))
            pa[:1500] += prefix_shift
            sig = np.round(pa / 0.1).astype(np.int16)
            w.add_read(pod5.Read(
                read_id=uuid.uuid4(), pore=pod5.Pore(channel=1, well=1, pore_type="r10"),
                calibration=cal, read_number=i, start_sample=0, median_before=100.0,
                end_reason=pod5.EndReason.from_reason_with_default_forced(pod5.EndReasonEnum.SIGNAL_POSITIVE),
                run_info=run_info, signal=sig))

def make(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{root}/human", exist_ok=True); os.makedirs(f"{root}/pathogen", exist_ok=True)
    runs = [("PAO89685", "aaaaaaaa", "SEBP48"), ("PAO89685", "bbbbbbbb", "SEBP48"),
            ("PAO98397", "cccccccc", "PC24"), ("PAO98633", "dddddddd", "SEBP48")]
    for k, (fcid, run, dev) in enumerate(runs):
        ri = _run_info(fcid, run, dev, "sqk-lsk114", "FLO-PRO114M")
        write_file(f"{root}/human/{fcid}_{run}.pod5", n, rng, ri, 90.0, 8.0)
    ri = _run_info("FAW09960", "eeeeeeee", "GXB", "sqk-rbk114-96", "FLO-MIN114")
    for j, sp in enumerate(["saureus", "ecoli", "paeruginosa", "kpneumoniae"]):
        write_file(f"{root}/pathogen/{sp}.pod5", n, rng, ri, 95.0 + j, 10.0, prefix_shift=20.0)
    return root
