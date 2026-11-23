"""
Comprehensive diagnostic script addressing User Questions 1 & 2:
1. Verify throughput metric: formula, units, raw ns-3 byte counters, logged Mbps.
2. Fixed-CW sweep: constant a = 4, 5, 6, 7, 8, 9, 10 (seeds 1, 2, 3) to determine ceiling.
"""
import subprocess
import os
import sys
import json
import time
import numpy as np
from ns3gym import ns3env

WAF_PWD = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
NS3_DIR = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29"

def run_ns3_episode(agent_type, dry_run, action_val, n_wifi=5, sim_time=5.0, seed=1, port=19000):
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": agent_type,
        "scenario": "basic",
        "nWifi": n_wifi,
        "dryRun": 1 if dry_run else 0,
        "seed": seed,
    }
    cmd = f'{WAF_PWD} --run "linear-mesh'
    for k, v in sim_args.items():
        cmd += f" --{k}={v}"
    cmd += f' --openGymPort={port}"'

    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        cwd=NS3_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    try:
        env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=seed, simArgs=sim_args, debug=False)
        obs = env.reset()
        steps = int(sim_time / 0.01)
        
        info_mbits_sum = 0.0
        step_throughputs = []
        cws = []
        per_sta_rx_approx = np.zeros(n_wifi, dtype=np.float64)

        # In continuous mode, cw.cc computes cw_val = pow(2, action + 4).
        # To test constant a in [4, 10], action = a - 4.0.
        # If action_val is a float:
        if isinstance(action_val, (int, float)):
            env_act = [float(action_val) - 4.0] * n_wifi
        else:
            env_act = action_val

        for s in range(steps):
            next_obs, rew, done, info = env.step(env_act)
            parts = str(info).split("|")
            mb_step = float(parts[0])  # info[0] is in Megabits
            info_mbits_sum += mb_step
            step_throughputs.append(mb_step / 0.01)
            if len(parts) > 1:
                cws.append(float(parts[1]))
            # Obs is frame loss rate per station
            per_sta_rx_approx += np.maximum(1.0 - np.array(next_obs).flatten(), 0.0)
            if done:
                break
        
        env.close()
        stdout_out, _ = proc.communicate(timeout=10)

        # Parse raw ns-3 final line
        # cw.cc line 570: printf("Sent mbytes: %.2f\tThroughput: %.3f", res, res/simulationTime);
        # res = final_total_rx * 1464 * 8.0 / 1024 / 1024 (in Megabits)
        ns3_raw_mbits = None
        ns3_raw_tput = None
        for line in stdout_out.splitlines():
            if "Sent mbytes:" in line and "Throughput:" in line:
                try:
                    p = line.split("Sent mbytes:")[1].split("Throughput:")
                    ns3_raw_mbits = float(p[0].strip())
                    ns3_raw_tput = float(p[1].strip())
                except:
                    pass

        # Fairness
        sum_x = np.sum(per_sta_rx_approx)
        sum_x2 = np.sum(per_sta_rx_approx**2)
        jain = float((sum_x**2) / (n_wifi * sum_x2)) if sum_x2 > 0 else 1.0

        mean_step_tput = float(np.mean(step_throughputs))
        sum_mbits_div_time = info_mbits_sum / sim_time

        return {
            "seed": seed,
            "action_a": action_val if isinstance(action_val, (int, float)) else "dynamic",
            "mean_cw": float(np.mean(cws)) if cws else (2**action_val if isinstance(action_val, (int, float)) else 0),
            "info_mbits_sum": round(info_mbits_sum, 4),
            "throughput_mbps": round(mean_step_tput, 3),
            "sum_div_time_mbps": round(sum_mbits_div_time, 3),
            "ns3_raw_mbits": ns3_raw_mbits,
            "ns3_raw_tput": ns3_raw_tput,
            "jain_fairness": round(jain, 4),
        }
    except Exception as e:
        proc.terminate()
        raise e

def sweep_fixed_cw():
    print("==================================================================")
    print("TASK 2: FIXED-CW SWEEP (a = 4, 5, 6, 7, 8, 9, 10; seeds 1, 2, 3)")
    print("==================================================================")
    actions = [4, 5, 6, 7, 8, 9, 10]
    seeds = [1, 2, 3]
    port_base = 18400

    results = {}
    port_counter = 0

    for a in actions:
        cw_val = 2**a
        results[a] = []
        print(f"\n--- Testing constant a = {a} (CW = 2^{a} = {cw_val}) ---")
        for s in seeds:
            port = port_base + port_counter
            port_counter += 1
            res = run_ns3_episode("continuous", dry_run=False, action_val=a, n_wifi=5, sim_time=5.0, seed=s, port=port)
            results[a].append(res)
            print(f"  Seed {s} -> Throughput: {res['throughput_mbps']:.2f} Mbps, Fairness: {res['jain_fairness']:.4f}, Raw ns3: {res['ns3_raw_tput']} Mb/s")
        
        tputs = [r["throughput_mbps"] for r in results[a]]
        fairs = [r["jain_fairness"] for r in results[a]]
        print(f"  >>> a = {a} (CW = {cw_val}) Mean Throughput: {np.mean(tputs):.2f} ± {np.std(tputs):.2f} Mbps | Mean Fairness: {np.mean(fairs):.4f}")

    # Also test BEB
    print("\n--- Testing Standard 802.11 (BEB; dryRun = 1) ---")
    beb_results = []
    for s in seeds:
        port = port_base + port_counter
        port_counter += 1
        res = run_ns3_episode("continuous", dry_run=True, action_val=4, n_wifi=5, sim_time=5.0, seed=s, port=port)
        beb_results.append(res)
        print(f"  BEB Seed {s} -> Throughput: {res['throughput_mbps']:.2f} Mbps, Fairness: {res['jain_fairness']:.4f}, Raw ns3: {res['ns3_raw_tput']} Mb/s")
    
    b_tputs = [r["throughput_mbps"] for r in beb_results]
    b_fairs = [r["jain_fairness"] for r in beb_results]
    print(f"  >>> BEB Mean Throughput: {np.mean(b_tputs):.2f} ± {np.std(b_tputs):.2f} Mbps | Mean Fairness: {np.mean(b_fairs):.4f}")

    out_file = "sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({"sweep": results, "beb": beb_results}, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    sweep_fixed_cw()
