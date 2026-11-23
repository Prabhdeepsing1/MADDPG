#!/usr/bin/env python3
import argparse
import csv
import statistics
import subprocess
import time
import numpy as np

from ns3gym import ns3env


WAF_PWD = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
NS3_DIR = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29"


def run_one(config, cw_value, n_wifi, sim_time, seed, port):
    is_beb = config == "beb"
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": "direct_continuous",
        "scenario": "basic",
        "nWifi": n_wifi,
        "dryRun": 1 if is_beb else 0,
        "seed": seed,
        "rng": seed,
    }
    cmd = f'{WAF_PWD} --run "linear-mesh'
    for key, value in sim_args.items():
        cmd += f" --{key}={value}"
    cmd += f' --openGymPort={port}"'

    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=NS3_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    env = ns3env.Ns3Env(
        port=port,
        stepTime=0.01,
        startSim=0,
        simSeed=seed,
        simArgs=sim_args,
        debug=False,
    )

    obs = env.reset()
    steps = int(sim_time / 0.01)
    action = [0.0 if is_beb else float(cw_value)] * n_wifi
    step_throughputs = []
    frame_losses = []
    rx_packets = [0] * n_wifi
    jain_episode = 1.0

    for _ in range(steps):
        next_obs, _, _, info = env.step(action)
        parts = str(info).split("|")
        mb_step = float(parts[0])
        step_throughputs.append(mb_step / 0.01)
        frame_losses.append(float(np.asarray(next_obs, dtype=np.float64).mean()))
        if len(parts) > 5:
            jain_episode = float(parts[5])
        if len(parts) > 6:
            rx_packets = [int(value) for value in parts[6].split(",")]

    env.close()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)

    median_rx = statistics.median(rx_packets) if rx_packets else 0.0
    threshold = median_rx * 0.5
    below = sum(value < threshold for value in rx_packets)

    return {
        "config": config,
        "cw": "beb" if is_beb else cw_value,
        "seed": seed,
        "mean_throughput_mbps": statistics.mean(step_throughputs),
        "measured_jain_episode": jain_episode,
        "frame_loss_rate": statistics.mean(frame_losses),
        "rx_min": min(rx_packets) if rx_packets else 0,
        "rx_median": median_rx,
        "rx_max": max(rx_packets) if rx_packets else 0,
        "stations_below_half_median": below,
    }


def summarize(rows):
    result = []
    for config in sorted(set(row["config"] for row in rows)):
        config_rows = [row for row in rows if row["config"] == config]
        cw_values = sorted(set(row["cw"] for row in config_rows), key=str)
        for cw in cw_values:
            selected = [row for row in config_rows if row["cw"] == cw]
            result.append({
                "config": config,
                "cw": cw,
                "throughput_mean": statistics.mean(r["mean_throughput_mbps"] for r in selected),
                "throughput_std": statistics.pstdev(r["mean_throughput_mbps"] for r in selected),
                "jain_mean": statistics.mean(r["measured_jain_episode"] for r in selected),
                "jain_std": statistics.pstdev(r["measured_jain_episode"] for r in selected),
                "frame_loss_mean": statistics.mean(r["frame_loss_rate"] for r in selected),
                "frame_loss_std": statistics.pstdev(r["frame_loss_rate"] for r in selected),
                "below_median_mean": statistics.mean(r["stations_below_half_median"] for r in selected),
                "below_median_std": statistics.pstdev(r["stations_below_half_median"] for r in selected),
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--sim-time", type=float, default=15.0)
    parser.add_argument("--n-wifi", type=int, default=40)
    parser.add_argument("--fixed-cw", nargs="+", type=int,
                        default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--csv", default="stage1_baselines.csv")
    args = parser.parse_args()

    port = 18000
    rows = []
    for cw in args.fixed_cw:
        for seed in args.seeds:
            print(f"Running fixed CW={cw}, seed={seed}")
            row = run_one("fixed", cw, args.n_wifi, args.sim_time, seed, port)
            port += 1
            rows.append(row)
            print(row)
    for seed in args.seeds:
        print(f"Running BEB, seed={seed}")
        row = run_one("beb", None, args.n_wifi, args.sim_time, seed, port)
        port += 1
        rows.append(row)
        print(row)

    fieldnames = list(rows[0].keys())
    with open(args.csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    print("\n=== STAGE 1 SUMMARY ===")
    for row in summary:
        print(
            f"{row['config']} cw={row['cw']} "
            f"throughput={row['throughput_mean']:.3f}+/-{row['throughput_std']:.3f} "
            f"jain={row['jain_mean']:.4f}+/-{row['jain_std']:.4f} "
            f"frame_loss={row['frame_loss_mean']:.4f}+/-{row['frame_loss_std']:.4f} "
            f"below_half_median={row['below_median_mean']:.2f}+/-{row['below_median_std']:.2f}"
        )


if __name__ == "__main__":
    main()
