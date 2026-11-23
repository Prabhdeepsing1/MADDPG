#!/usr/bin/env python3
"""Step 4 benchmark: legacy vs centralized MADDPG, identical settings.

Runs maddpg_train.py with simTime=15, seeds 1-5, for both
--use_centralized_critic and legacy (no flag). Same start_timesteps/buffer.

Usage:  python3 run_step4_bench.py [--episodes 12] [--seeds 1 2 3 4 5]
"""
import argparse
import json
import os
import subprocess
import sys

WANDB_OFFLINE = "offline"


def run_one(variant, seed, episodes, sim_time, n_wifi, start_timesteps, batch_size,
            buffer_size, discount, out_dir):
    cmd = [
        sys.executable, "maddpg_train.py",
        "--nWifi", str(n_wifi),
        "--simTime", str(sim_time),
        "--stepTime", "0.01",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--start_timesteps", str(start_timesteps),
        "--batch_size", str(batch_size),
        "--buffer_size", str(buffer_size),
        "--discount", str(discount),
    ]
    if variant == "centralized":
        cmd.append("--use_centralized_critic")
    tag = f"{variant}_seed{seed}"
    jsonl = os.path.join(out_dir, f"{tag}.jsonl")
    summary = os.path.join(out_dir, f"{tag}.json")
    cmd += ["--jsonl", jsonl, "--summary_json", summary]

    env = os.environ.copy()
    env["WANDB_MODE"] = WANDB_OFFLINE

    log = os.path.join(out_dir, f"{tag}.log")
    print(f"\n{'='*60}\nRUN {tag}\n{'='*60}")
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        for line in proc.stdout:
            lf.write(line)
            if "Mean Throughput:" in line or "Error" in line or "Traceback" in line:
                print("  " + line.rstrip())
        proc.wait()
    if proc.returncode != 0:
        print(f"  !! {tag} exited with code {proc.returncode}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--simTime", type=float, default=15)
    ap.add_argument("--nWifi", type=int, default=5)
    ap.add_argument("--start_timesteps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--buffer_size", type=int, default=100000)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--variants", nargs="+", default=["legacy", "centralized"],
                    choices=["legacy", "centralized"])
    ap.add_argument("--out_dir", default="bench_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for variant in args.variants:
        for seed in args.seeds:
            s = run_one(variant, seed, args.episodes, args.simTime, args.nWifi,
                        args.start_timesteps, args.batch_size, args.buffer_size,
                        args.discount, args.out_dir)
            with open(s) as f:
                data = json.load(f)
            eps = data["episodes"]
            mean_tput = data["final_mean_throughput_mbps"]
            last_ep = eps[-1] if eps else {}
            results.append({
                "variant": variant, "seed": seed,
                "episodes": [e["mean_throughput_mbps"] for e in eps],
                "mean_throughput": round(mean_tput, 2),
                "last_ep_throughput": round(last_ep.get("mean_throughput_mbps", 0), 2),
                "last_ep_action": round(last_ep.get("mean_action", 0), 3),
            })

    # Paired table
    import statistics
    print("\n\n" + "=" * 70)
    print("STEP 4 BENCHMARK TABLE (simTime=%s, start_timesteps=%s, batch=%s, γ=%s)"
          % (args.simTime, args.start_timesteps, args.batch_size, args.discount))
    print("=" * 70)
    print(f"{'variant':<14}{'seed':>5}{'mean_tput':>11}{'last_ep':>9}{'last_act':>10}")
    for r in results:
        print(f"{r['variant']:<14}{r['seed']:>5}{r['mean_throughput']:>11.2f}"
              f"{r['last_ep_throughput']:>9.2f}{r['last_ep_action']:>10.3f}")

    for variant in args.variants:
        rs = [r for r in results if r["variant"] == variant]
        if rs:
            ts = [r["mean_throughput"] for r in rs]
            ls = [r["last_ep_throughput"] for r in rs]
            acts = [r["last_ep_action"] for r in rs]
            print(f"\n{variant}: mean_tput = {statistics.mean(ts):.2f} ± {statistics.stdev(ts) if len(ts) > 1 else 0:.2f} Mbps")
            print(f"{variant}: last_ep_tput = {statistics.mean(ls):.2f} ± {statistics.stdev(ls) if len(ls) > 1 else 0:.2f} Mbps")
            print(f"{variant}: last_ep_action = {statistics.mean(acts):.3f} ± {statistics.stdev(acts) if len(acts) > 1 else 0:.3f}")

    out = os.path.join(args.out_dir, "bench_summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()