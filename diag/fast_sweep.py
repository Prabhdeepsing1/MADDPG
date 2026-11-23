"""
diag/fast_sweep.py
Run static CW sweep {15, 31, 63, 127, 255, 383, 511, 1023} in parallel batches.
"""
import time, subprocess, sys
from multiprocessing import Pool
import numpy as np
from ns3gym import ns3env

def evaluate_cw(args):
    cw_val, port, sim_time, seed = args
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": "direct_continuous",
        "scenario": "basic",
        "nWifi": 40,
        "dryRun": 0,
        "seed": seed,
    }
    waf_pwd = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
    cmd = f'{waf_pwd} --run "linear-mesh'
    for k, v in sim_args.items():
        cmd += f" --{k}={v}"
    cmd += f' --openGymPort={port}"'
    
    proc = subprocess.Popen(['bash', '-c', cmd], cwd="/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29")
    try:
        env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=seed, simArgs=sim_args, debug=False)
        obs = env.reset()
        steps = int(sim_time / 0.01)
        action = [float(cw_val)] * 40
        total_mb = 0.0
        collision_rates = []
        per_sta_rx = np.zeros(40, dtype=np.float64)
        
        for s in range(steps):
            no, rew, dn, inf = env.step(action)
            parts = inf.split("|")
            mb = float(parts[0])
            total_mb += mb
            collision_rates.append(np.mean(no))
            # approximate per-station success rate
            per_sta_rx += np.maximum(1.0 - np.array(no).flatten(), 0.0)
            if dn:
                break
        env.close()
        throughput = total_mb / sim_time
        mean_coll = float(np.mean(collision_rates))
        
        # Jain's fairness index
        sum_x = np.sum(per_sta_rx)
        sum_x2 = np.sum(per_sta_rx**2)
        jain = float((sum_x**2) / (40 * sum_x2)) if sum_x2 > 0 else 1.0
        
        return {
            "cw": cw_val,
            "throughput_mbps": throughput,
            "collision_prob": mean_coll,
            "jain_index": jain,
            "seed": seed,
        }
    finally:
        proc.terminate()

def run_standard_80211(args):
    port, sim_time, seed = args
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": "continuous",
        "scenario": "basic",
        "nWifi": 40,
        "dryRun": 1,
        "seed": seed,
    }
    waf_pwd = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
    cmd = f'{waf_pwd} --run "linear-mesh'
    for k, v in sim_args.items():
        cmd += f" --{k}={v}"
    cmd += f' --openGymPort={port}"'
    
    proc = subprocess.Popen(['bash', '-c', cmd], cwd="/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29")
    try:
        env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=seed, simArgs=sim_args, debug=False)
        obs = env.reset()
        steps = int(sim_time / 0.01)
        action = [0.0] * 40
        total_mb = 0.0
        collision_rates = []
        per_sta_rx = np.zeros(40, dtype=np.float64)
        
        for s in range(steps):
            no, rew, dn, inf = env.step(action)
            parts = inf.split("|")
            mb = float(parts[0])
            total_mb += mb
            collision_rates.append(np.mean(no))
            per_sta_rx += np.maximum(1.0 - np.array(no).flatten(), 0.0)
            if dn:
                break
        env.close()
        throughput = total_mb / sim_time
        mean_coll = float(np.mean(collision_rates))
        sum_x = np.sum(per_sta_rx)
        sum_x2 = np.sum(per_sta_rx**2)
        jain = float((sum_x**2) / (40 * sum_x2)) if sum_x2 > 0 else 1.0
        return {
            "type": "Standard 802.11 (BEB)",
            "throughput_mbps": throughput,
            "collision_prob": mean_coll,
            "jain_index": jain,
            "seed": seed,
        }
    finally:
        proc.terminate()

if __name__ == "__main__":
    t0 = time.time()
    cw_list = [15, 31, 63, 127, 255, 383, 511, 1023]
    sim_time = 5.0
    
    print("Running parallel sweep for 8 CW values...")
    # 8 tasks on distinct ports
    tasks = [(cw, 16000 + i, sim_time, 1) for i, cw in enumerate(cw_list)]
    
    with Pool(processes=4) as pool:
        results = pool.map(evaluate_cw, tasks)
        
    print("\n=== Static CW Sweep Results ===")
    print(f"{'CW':<6} | {'Throughput (Mb/s)':<18} | {'Collision Prob':<15} | {'Jain Index':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['cw']:<6} | {r['throughput_mbps']:<18.2f} | {r['collision_prob']:<15.3f} | {r['jain_index']:<10.3f}")
        
    # Also run standard 802.11 across 3 seeds:
    print("\nRunning Standard 802.11 across 3 seeds...")
    beb_tasks = [(17000 + i, sim_time, s) for i, s in enumerate([0, 1, 2])]
    with Pool(processes=3) as pool:
        beb_results = pool.map(run_standard_80211, beb_tasks)
        
    print("\n=== Standard 802.11 (BEB) Results ===")
    print(f"{'Seed':<6} | {'Throughput (Mb/s)':<18} | {'Collision Prob':<15} | {'Jain Index':<10}")
    print("-" * 55)
    for r in beb_results:
        print(f"{r['seed']:<6} | {r['throughput_mbps']:<18.2f} | {r['collision_prob']:<15.3f} | {r['jain_index']:<10.3f}")
        
    tputs = [r['throughput_mbps'] for r in beb_results]
    colls = [r['collision_prob'] for r in beb_results]
    jains = [r['jain_index'] for r in beb_results]
    print("-" * 55)
    print(f"Mean   | {np.mean(tputs):<18.2f} | {np.mean(colls):<15.3f} | {np.mean(jains):<10.3f}")
    print(f"Std    | {np.std(tputs):<18.2f} | {np.std(colls):<15.3f} | {np.std(jains):<10.3f}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")

