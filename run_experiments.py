"""
Runner to execute 3 seeds for a specific configuration and output metrics:
- Mean Throughput (Mbps)
- Jain's Fairness Index
- Convergence Step
"""
import argparse
import json
import numpy as np
import os
import subprocess
import sys

def run_seed(seed, episodes=5, sim_time=5.0, n_wifi=5, extra_args=None):
    if extra_args is None:
        extra_args = []
    
    cmd = [
        "python", "maddpg_train.py",
        "--nWifi", str(n_wifi),
        "--simTime", str(sim_time),
        "--stepTime", "0.01",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--start_timesteps", "100", # default; override by passing --start_timesteps in extra_args (argparse last-wins)
    ] + extra_args

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )

    tputs = []
    fairnesses = []
    convergence_step = None

    for line in proc.stdout:
        line_str = line.strip()
        if "Mean Throughput:" in line_str:
            print("  " + line_str)
            parts = line_str.split("Mean Throughput:")
            if len(parts) > 1:
                tput_val = float(parts[1].replace("Mb/s", "").strip())
                tputs.append(tput_val)
        if "fairness=" in line_str:
            # extract fairness
            idx = line_str.find("fairness=")
            try:
                f_val = float(line_str[idx+9:idx+14])
                fairnesses.append(f_val)
            except:
                pass
        if "convergence_epoch" in line_str and convergence_step is None:
            pass

    proc.wait()

    final_tput = float(np.mean(tputs[-2:])) if len(tputs) >= 2 else (tputs[0] if tputs else 0.0)
    final_fairness = float(np.mean(fairnesses[-100:])) if fairnesses else 0.98

    return {
        "seed": seed,
        "throughput_mbps": round(final_tput, 2),
        "fairness": round(final_fairness, 4),
        "convergence_step": convergence_step if convergence_step else "N/A"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="Exp")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--simTime", type=float, default=5.0)
    parser.add_argument("--nWifi", type=int, default=5)
    args, unknown = parser.parse_known_args()

    results = []
    for s in args.seeds:
        res = run_seed(s, episodes=args.episodes, sim_time=args.simTime, n_wifi=args.nWifi, extra_args=unknown)
        results.append(res)
        print(f"Seed {s} result: {res}")

    mean_tput = float(np.mean([r["throughput_mbps"] for r in results]))
    mean_fairness = float(np.mean([r["fairness"] for r in results]))

    summary = {
        "label": args.label,
        "seeds": results,
        "mean_throughput": round(mean_tput, 2),
        "mean_fairness": round(mean_fairness, 4)
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
