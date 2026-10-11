"""
ReadUntil (adaptive sampling) simulator for FreeSolo.

Revised Oct 2026. The original simulator (i) compared against a "baseline"
that was not a real no-ejection run, which produced a negative
"bases saved"; (ii) used invented read-length distributions; (iii) counted
enrichment in reads rather than bases; and (iv) ignored capture and unblock
time. This version simulates two runs on the same random stream:

  * control  : every captured strand is sequenced to completion
  * adaptive : after `decision_samples` of signal (offset + window(s)) plus
               compute latency, strands classified as human are unblocked

and reports enrichment the way adaptive-sampling papers do:

  * relative enrichment  = pathogen share of sequenced bases (adaptive)
                           / pathogen share of sequenced bases (control)
  * absolute enrichment  = pathogen bases (adaptive) / pathogen bases (control)
                           (yield gain; > 1 only if ejection frees enough
                           pore time to outweigh false ejections)
  * read-level enrichment (the metric used in the first submission) is still
    reported for comparison.

Decisions are drawn from classifier probabilities measured on real held-out
reads; strand lengths are drawn from empirical read-length distributions.
All time quantities are in seconds; lengths in raw samples (5 kHz).
Bases are approximated as samples / SAMPLES_PER_BASE (400 b/s at 5 kHz).
"""

import numpy as np
import json
import os
from datetime import datetime

SAMPLE_RATE = 5000
SAMPLES_PER_BASE = SAMPLE_RATE / 400.0   # R10.4.1 e8.2 400 bps


class ReadUntilSimulator:
    def __init__(self, n_pores=512, human_fraction=0.99, capture_s=0.5,
                 unblock_s=0.1, seed=42, classifier=None, **_ignored):
        """
        n_pores       : channels (512 on a MinION/GridION flow cell)
        capture_s     : mean idle time before a pore captures a new strand
        unblock_s     : time to eject a strand and return to idle
        classifier    : optional FreeSoloClassifier, only needed for the
                        run_simulation_from_arrays convenience wrapper
        """
        self.n_pores = n_pores
        self.human_fraction = human_fraction
        self.capture_s = capture_s
        self.unblock_s = unblock_s
        self.seed = seed
        self.classifier = classifier

    # ------------------------------------------------------------------ core
    def _run(self, adaptive, p_h, p_p, len_h, len_p, decision_samples,
             latency_s, threshold, duration_s):
        rng = np.random.RandomState(self.seed)   # identical stream for both runs
        fs = SAMPLE_RATE
        c = dict(h_captured=0, p_captured=0, h_ejected=0, h_missed=0,
                 p_ejected=0, p_kept=0, h_samples=0.0, p_samples=0.0,
                 decided=0, too_short=0)
        for _ in range(self.n_pores):
            t = 0.0
            while True:
                t += rng.exponential(self.capture_s)
                if t >= duration_s:
                    break
                is_h = rng.random_sample() < self.human_fraction
                L = float((len_h if is_h else len_p)[rng.randint(len(len_h if is_h else len_p))])
                prob = (p_h if is_h else p_p)[rng.randint(len(p_h if is_h else p_p))]
                c["h_captured" if is_h else "p_captured"] += 1
                seq = L
                if adaptive and L > decision_samples:
                    c["decided"] += 1
                    eject = prob < threshold
                    if eject:
                        seq = decision_samples
                        dt = decision_samples / fs + latency_s + self.unblock_s
                        c["h_ejected" if is_h else "p_ejected"] += 1
                    else:
                        dt = L / fs
                        c["h_missed" if is_h else "p_kept"] += 1
                else:
                    if adaptive:
                        c["too_short"] += 1
                    dt = L / fs
                if t + dt > duration_s:           # truncate at end of run
                    seq = min(seq, (duration_s - t) * fs)
                    dt = duration_s - t
                c["h_samples" if is_h else "p_samples"] += seq
                t += dt
        return c

    def run_simulation_from_probs(self, p_human, p_pathogen, len_human, len_pathogen,
                                  decision_samples, latency_ms=0.0, threshold=0.5,
                                  duration_seconds=3600, verbose=True):
        """p_human / p_pathogen: P(pathogen) for real held-out human / pathogen
        reads. len_*: empirical raw read lengths (samples)."""
        p_h, p_p = np.asarray(p_human, float), np.asarray(p_pathogen, float)
        len_h, len_p = np.asarray(len_human, float), np.asarray(len_pathogen, float)
        lat = latency_ms / 1000.0
        ctrl = self._run(False, p_h, p_p, len_h, len_p, decision_samples, lat, threshold, duration_seconds)
        ada = self._run(True, p_h, p_p, len_h, len_p, decision_samples, lat, threshold, duration_seconds)

        def share(c):
            tot = c["h_samples"] + c["p_samples"]
            return c["p_samples"] / tot if tot else 0.0
        b = SAMPLES_PER_BASE
        kept = ada["p_kept"] + ada["h_missed"]
        orig_read_pct = ctrl["p_captured"] / max(1, ctrl["p_captured"] + ctrl["h_captured"])
        read_level = (ada["p_kept"] / kept) / orig_read_pct if kept and orig_read_pct else float("nan")
        results = {
            "simulation_params": {
                "n_pores": self.n_pores, "human_fraction": self.human_fraction,
                "duration_seconds": duration_seconds, "capture_s": self.capture_s,
                "unblock_s": self.unblock_s, "decision_samples": int(decision_samples),
                "decision_time_s": decision_samples / SAMPLE_RATE + lat,
                "latency_ms": latency_ms, "threshold": threshold,
                "sampling_rate_hz": SAMPLE_RATE, "samples_per_base": b,
                "mean_len_human_bases": float(len_h.mean() / b),
                "mean_len_pathogen_bases": float(len_p.mean() / b),
                "seed": self.seed,
            },
            "control": {k: float(v) for k, v in ctrl.items()},
            "adaptive": {k: float(v) for k, v in ada.items()},
            "bases": {
                "control_pathogen_bases": ctrl["p_samples"] / b,
                "adaptive_pathogen_bases": ada["p_samples"] / b,
                "control_total_bases": (ctrl["h_samples"] + ctrl["p_samples"]) / b,
                "adaptive_total_bases": (ada["h_samples"] + ada["p_samples"]) / b,
                "control_pathogen_share": share(ctrl),
                "adaptive_pathogen_share": share(ada),
            },
            "enrichment": {
                "relative_enrichment_bases": share(ada) / share(ctrl) if share(ctrl) else float("nan"),
                "absolute_enrichment_bases": ada["p_samples"] / ctrl["p_samples"] if ctrl["p_samples"] else float("nan"),
                "read_level_enrichment": read_level,
            },
            "classification": {
                "human_ejected": ada["h_ejected"], "human_missed": ada["h_missed"],
                "pathogen_kept": ada["p_kept"], "pathogen_ejected": ada["p_ejected"],
                "strands_too_short_to_decide": ada["too_short"],
            },
            "timestamp": datetime.now().isoformat(),
        }
        if verbose:
            e = results["enrichment"]
            print(f"  Relative enrichment (bases): {e['relative_enrichment_bases']:.2f}x  "
                  f"Absolute (pathogen yield): {e['absolute_enrichment_bases']:.2f}x  "
                  f"Read-level: {e['read_level_enrichment']:.2f}x")
        return results

    # ------------------------------------------------------- compatibility
    def run_simulation_from_arrays(self, X_test, y_test, duration_seconds=3600,
                                   len_human=None, len_pathogen=None,
                                   offset=1500, latency_ms=None, verbose=True):
        """Wrapper used by run_pipeline.py: scores the held-out windows with
        self.classifier, then calls run_simulation_from_probs."""
        if self.classifier is None:
            raise ValueError("classifier required")
        proba = self.classifier.predict_proba(X_test)[:, 1]
        window = X_test.shape[1]
        if latency_ms is None:
            lat = [self.classifier.predict_single(X_test[i])[2] for i in range(min(100, len(X_test)))]
            latency_ms = float(np.mean(lat))
        if len_human is None or len_pathogen is None:
            raise ValueError("Empirical read lengths (len_human, len_pathogen) are required; "
                             "take them from the 'raw_len' column of the read table.")
        return self.run_simulation_from_probs(
            proba[y_test == 0], proba[y_test == 1], len_human, len_pathogen,
            decision_samples=offset + window, latency_ms=latency_ms,
            duration_seconds=duration_seconds, verbose=verbose)

    def save_results(self, results, output_dir, name="simulation_results.json"):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(results, f, indent=2, default=str)
