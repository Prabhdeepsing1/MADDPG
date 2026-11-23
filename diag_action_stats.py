#!/usr/bin/env python3
"""Step 4 Task 3 — Action diagnostics.

Reads per-step JSONL logs (obs + joint_action) for OSCAR DDPG, legacy
MADDPG and centralized MADDPG (seed 1) and reports per-epoch:
  - mean action a across stations and time
  - std of actions across stations (do actors diverge?)
  - whether actions saturate at the [4, 10] boundary
"""
import json
import sys
import numpy as np


def load(path):
    obs, act = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            obs.append(r["obs"])
            act.append(r["joint_action"])
    return np.array(obs), np.array(act)


def analyze(name, path, n_epochs=12, steps_per_ep=1500):
    obs, act = load(path)
    T, N = act.shape
    print(f"\n=== {name} ({path}) ===")
    print(f"  T={T} steps, N={N} stations")

    # Per-station action statistics over the whole run
    per_sta_mean = act.mean(axis=0)
    per_sta_std = act.std(axis=0)
    print(f"  per-station mean action: {np.round(per_sta_mean, 3)}")
    print(f"  per-station std  action: {np.round(per_sta_std, 3)}")

    # Saturation at boundary [4, 10]
    frac_at_4 = float((act <= 4.001).mean())
    frac_at_10 = float((act >= 9.999).mean())
    print(f"  fraction of actions at a=4 (CW=16):  {frac_at_4:.3f}")
    print(f"  fraction of actions at a=10 (CW=1024): {frac_at_10:.3f}")

    # Std across stations per step (divergence of actors)
    cross_std = act.std(axis=1)
    print(f"  mean cross-station action std: {cross_std.mean():.4f} "
          f"(0 => all actors identical)")

    # Per-epoch stats
    print("  epoch | mean_a | std_across_sta | mean_tput | final50_mean_a")
    rows = []
    for e in range(n_epochs):
        sl = act[e * steps_per_ep:(e + 1) * steps_per_ep]
        if len(sl) == 0:
            break
        obs_sl = obs[e * steps_per_ep:(e + 1) * steps_per_ep]
        # throughput from obs is not available here; use reward/throughput field
        rows.append({
            "epoch": e,
            "mean_a": float(sl.mean()),
            "std_across_sta": float(sl.std()),
            "mean_obs": float(obs_sl.mean()) if len(obs_sl) else 0.0,
        })
        print(f"    {e:2d}  | {sl.mean():6.3f} | {sl.std():10.4f} | "
              f"obs={obs_sl.mean():.4f}")
    return rows


if __name__ == "__main__":
    # paths are passed as name=path pairs
    args = sys.argv[1:]
    for a in args:
        name, path = a.split("=", 1)
        analyze(name, path)
    print("\nNote: 'std_across_sta' > 0 means actors chose DIFFERENT CWs per station;")
    print("      'std_across_sta' ~ 0 means all actors converged to the same value.")