#!/usr/bin/env python3
"""Step 4 Task 2 — Observation check.

Reads per-step JSONL logs from the benchmark and reports whether the
per-station observations differ across stations or are identical.

If observations are identical across stations, N separate actors have no
information advantage over a single shared actor.
"""
import json
import sys
import numpy as np


def analyze(jsonl_path):
    obs_rows = []
    act_rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            obs_rows.append(r["obs"])
            act_rows.append(r["joint_action"])
    obs = np.array(obs_rows)      # (T, N)
    act = np.array(act_rows)      # (T, N)

    T, N = obs.shape
    # Per-step spread across stations
    per_step_spread = obs.std(axis=1)          # (T,)
    mean_spread = float(per_step_spread.mean())
    max_spread = float(per_step_spread.max())

    # Station-mean vs station-variance
    station_means = obs.mean(axis=0)           # (N,)
    station_stds = obs.std(axis=0)             # (N,)

    # Correlation between stations' observation streams
    corr = np.corrcoef(obs.T) if T > 1 else np.eye(N)
    offdiag = corr[np.triu_indices(N, 1)]

    print(f"File: {jsonl_path}")
    print(f"  steps T={T}, stations N={N}")
    print(f"  per-step std-across-stations: mean={mean_spread:.6f}, max={max_spread:.6f}")
    print(f"  station means: {np.round(station_means, 4)}")
    print(f"  station stds : {np.round(station_stds, 4)}")
    print(f"  cross-station obs correlation: mean={offdiag.mean():.4f}, min={offdiag.min():.4f}, max={offdiag.max():.4f}")
    print(f"  action spread (std across stations): mean={act.std(axis=1).mean():.4f}")
    if mean_spread < 1e-6 and max_spread < 1e-6:
        print("  -> OBSERVATIONS ARE IDENTICAL ACROSS STATIONS: N actors have no information advantage.")
    else:
        print("  -> Observations DIFFER across stations: per-station actors have distinct information.")
    return mean_spread


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
        print()