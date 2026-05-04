"""
Simulated ReadUntil Hardware Feedback (Could be exchanged for actual hardware)

Simulates MinION flow cell with autonomous eject/keep decisions.
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from collections import defaultdict


class SimulatedPore:
    def __init__(self, pore_id, rng):
        self.pore_id = pore_id
        self.state = 'idle'
        self.current_strand_type = None
        self.samples_read = 0
        self.strand_length = 0
        self.rng = rng

    def capture_strand(self, human_fraction=0.99):
        self.state = 'sequencing'
        self.samples_read = 0
        if self.rng.random() < human_fraction:
            self.current_strand_type = 'human'
            self.strand_length = max(2000, int(self.rng.exponential(20000)))
        else:
            self.current_strand_type = 'pathogen'
            self.strand_length = max(2000, int(self.rng.exponential(8000)))

    def unblock(self): self.state = 'unblocking'; self.current_strand_type = None
    def recover(self): self.state = 'recovering'
    def make_idle(self): self.state = 'idle'; self.current_strand_type = None; self.samples_read = 0


class ReadUntilSimulator:
    def __init__(self, classifier, data_generator, n_pores=256,
                 human_fraction=0.99, seed=42, human_fasta=None, pathogen_fasta=None):
        self.classifier = classifier
        self.data_generator = data_generator
        self.n_pores = n_pores
        self.human_fraction = human_fraction
        self.human_fasta = human_fasta
        self.pathogen_fasta = pathogen_fasta
        self.rng = np.random.RandomState(seed)
        self.pores = [SimulatedPore(i, self.rng) for i in range(n_pores)]
        self.window_size = 2000
        self.window_time_ms = (self.window_size / 4000) * 1000

    def run_simulation(self, duration_seconds=10, verbose=True):
        if verbose:
            print(f"\n{'='*60}\nVSI ReadUntil Simulation\n{'='*60}")
            print(f"  Pores: {self.n_pores}, Human: {self.human_fraction*100:.0f}%")

        for p in self.pores: p.make_idle()
        n_steps = int(duration_seconds / (self.window_time_ms / 1000))
        total_strands = human_strands = pathogen_strands = 0
        human_unblocked = human_missed = pathogen_ejected = pathogen_sequenced = 0
        total_bases_vsi = total_bases_baseline = 0
        latencies = []

        for step in range(n_steps):
            for pore in self.pores:
                if pore.state == 'idle':
                    pore.capture_strand(self.human_fraction)
                    total_strands += 1
                    if pore.current_strand_type == 'human': human_strands += 1
                    else: pathogen_strands += 1
                elif pore.state == 'sequencing':
                    pore.samples_read += self.window_size
                    if pore.samples_read == self.window_size:
                        sig = (self.data_generator.generate_human_signal(self.human_fasta, self.window_size)
                               if pore.current_strand_type == 'human'
                               else self.data_generator.generate_pathogen_signal(self.pathogen_fasta, self.window_size))
                        pred, conf, lat = self.classifier.predict_single(sig)
                        latencies.append(lat)
                        if pred == 0:
                            if pore.current_strand_type == 'human':
                                human_unblocked += 1
                            else:
                                pathogen_ejected += 1
                            total_bases_vsi += self.window_size
                            pore.unblock()
                        else:
                            if pore.current_strand_type == 'pathogen': pass
                            else: human_missed += 1
                    if pore.samples_read >= pore.strand_length:
                        total_bases_vsi += pore.strand_length
                        if pore.current_strand_type == 'pathogen': pathogen_sequenced += 1
                        pore.make_idle()
                    total_bases_baseline += self.window_size
                elif pore.state == 'unblocking': pore.recover()
                elif pore.state == 'recovering': pore.make_idle()

            if verbose and (step+1) % (n_steps//10) == 0:
                print(f"  {(step+1)/n_steps*100:.0f}% | Strands: {total_strands} | "
                      f"Unblocked: {human_unblocked} | Pathogens: {pathogen_sequenced}")

        orig_pct = pathogen_strands / max(1, total_strands) * 100
        kept = pathogen_sequenced + human_missed
        enriched_pct = pathogen_sequenced / max(1, kept) * 100
        enrichment = enriched_pct / max(0.01, orig_pct)

        results = {
            'simulation_params': {'n_pores': self.n_pores, 'human_fraction': self.human_fraction,
                                  'duration_seconds': duration_seconds},
            'strand_counts': {'total': total_strands, 'human': human_strands, 'pathogen': pathogen_strands},
            'classification': {
                'human_correctly_unblocked': human_unblocked, 'human_missed': human_missed,
                'pathogen_correctly_sequenced': pathogen_sequenced,
                'pathogen_incorrectly_ejected': pathogen_ejected,
                'unblock_accuracy': human_unblocked / max(1, human_unblocked + human_missed),
            },
            'efficiency': {
                'bases_sequenced_vsi': total_bases_vsi, 'bases_sequenced_baseline': total_bases_baseline,
                'bases_saved': total_bases_baseline - total_bases_vsi,
                'percentage_saved': (total_bases_baseline - total_bases_vsi) / max(1, total_bases_baseline) * 100,
            },
            'enrichment': {'original_pathogen_pct': orig_pct, 'enriched_pathogen_pct': enriched_pct,
                          'enrichment_factor': enrichment},
            'latency': {
                'mean_ms': float(np.mean(latencies)) if latencies else 0,
                'median_ms': float(np.median(latencies)) if latencies else 0,
                'p95_ms': float(np.percentile(latencies, 95)) if latencies else 0,
            },
            'timestamp': datetime.now().isoformat(),
        }
        if verbose:
            print(f"\n  Human unblocked: {human_unblocked}, Pathogen kept: {pathogen_sequenced}")
            print(f"  False ejections: {pathogen_ejected}, Missed human: {human_missed}")
            print(f"  Enrichment: {enrichment:.1f}x")
        return results

    def run_simulation_from_arrays(self, X_test, y_test, duration_seconds=10, verbose=True):
        """
        ReadUntil simulation using real held-out test signals.
        Draws actual signals from X_test instead of synthetic generators.
        This matches the paper's description: signals are drawn from the real
        held-out test set rather than synthetic generators.
        """
        if verbose:
            print(f"\n{'='*60}\nFreeSolo ReadUntil Simulation (real signals)\n{'='*60}")
            print(f"  Pores: {self.n_pores}, Human background: {self.human_fraction*100:.0f}%")

        # Pre-separate test signals by class for fast sampling
        human_signals    = X_test[y_test == 0]
        pathogen_signals = X_test[y_test == 1]
        if len(human_signals) == 0 or len(pathogen_signals) == 0:
            raise ValueError("X_test must contain both human (0) and pathogen (1) reads.")

        window_size   = X_test.shape[1]
        window_time_s = window_size / 4000.0   # 4 kHz sampling
        n_steps       = int(duration_seconds / window_time_s)

        for p in self.pores:
            p.make_idle()

        total_strands = human_strands = pathogen_strands = 0
        human_unblocked = human_missed = pathogen_ejected = pathogen_sequenced = 0
        total_bases_vsi = total_bases_baseline = 0
        latencies = []

        h_idx = 0
        p_idx = 0

        for step in range(n_steps):
            for pore in self.pores:
                if pore.state == 'idle':
                    pore.capture_strand(self.human_fraction)
                    total_strands += 1
                    if pore.current_strand_type == 'human':
                        human_strands += 1
                    else:
                        pathogen_strands += 1

                elif pore.state == 'sequencing':
                    pore.samples_read += window_size
                    if pore.samples_read == window_size:
                        # Draw real signal for this pore's strand type
                        if pore.current_strand_type == 'human':
                            sig = human_signals[h_idx % len(human_signals)]
                            h_idx += 1
                        else:
                            sig = pathogen_signals[p_idx % len(pathogen_signals)]
                            p_idx += 1

                        pred, conf, lat = self.classifier.predict_single(sig)
                        latencies.append(lat)

                        if pred == 0:   # classified as human -> eject
                            if pore.current_strand_type == 'human':
                                human_unblocked += 1
                            else:
                                pathogen_ejected += 1
                            total_bases_vsi += window_size
                            pore.unblock()
                        else:           # classified as pathogen -> keep sequencing
                            if pore.current_strand_type == 'human':
                                human_missed += 1

                    if pore.samples_read >= pore.strand_length:
                        total_bases_vsi += pore.strand_length
                        if pore.current_strand_type == 'pathogen':
                            pathogen_sequenced += 1
                        pore.make_idle()

                    total_bases_baseline += window_size

                elif pore.state == 'unblocking':
                    pore.recover()
                elif pore.state == 'recovering':
                    pore.make_idle()

            if verbose and n_steps >= 10 and (step + 1) % (n_steps // 10) == 0:
                print(f"  {(step+1)/n_steps*100:.0f}% | Strands: {total_strands} | "
                      f"Unblocked: {human_unblocked} | Pathogens kept: {pathogen_sequenced}")

        orig_pct     = pathogen_strands / max(1, total_strands) * 100
        kept         = pathogen_sequenced + human_missed
        enriched_pct = pathogen_sequenced / max(1, kept) * 100
        enrichment   = enriched_pct / max(0.01, orig_pct)

        results = {
            'simulation_params': {
                'n_pores': self.n_pores,
                'human_fraction': self.human_fraction,
                'duration_seconds': duration_seconds,
                'signal_source': 'real_held_out_test_set',
            },
            'strand_counts': {
                'total': total_strands, 'human': human_strands, 'pathogen': pathogen_strands,
            },
            'classification': {
                'human_correctly_unblocked': human_unblocked,
                'human_missed': human_missed,
                'pathogen_correctly_sequenced': pathogen_sequenced,
                'pathogen_incorrectly_ejected': pathogen_ejected,
                'unblock_accuracy': human_unblocked / max(1, human_unblocked + human_missed),
            },
            'efficiency': {
                'bases_sequenced_vsi': total_bases_vsi,
                'bases_sequenced_baseline': total_bases_baseline,
                'bases_saved': total_bases_baseline - total_bases_vsi,
                'percentage_saved': (total_bases_baseline - total_bases_vsi) / max(1, total_bases_baseline) * 100,
            },
            'enrichment': {
                'original_pathogen_pct': orig_pct,
                'enriched_pathogen_pct': enriched_pct,
                'enrichment_factor': enrichment,
            },
            'latency': {
                'mean_ms':   float(np.mean(latencies))           if latencies else 0,
                'median_ms': float(np.median(latencies))         if latencies else 0,
                'p95_ms':    float(np.percentile(latencies, 95)) if latencies else 0,
            },
            'timestamp': datetime.now().isoformat(),
        }

        if verbose:
            print(f"\n  Human unblocked:   {human_unblocked}")
            print(f"  Pathogen kept:     {pathogen_sequenced}")
            print(f"  False ejections:   {pathogen_ejected}")
            print(f"  Missed human:      {human_missed}")
            print(f"  Enrichment factor: {enrichment:.1f}x")

        return results

    def save_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
