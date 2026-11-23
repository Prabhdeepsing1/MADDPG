"""
diag/run_beb_seeds.py
Run standard 802.11 BEB across 3 valid seeds (1, 2, 3)
"""
import time, subprocess
import numpy as np
from ns3gym import ns3env

def run_standard_80211(seed, port=18100, sim_time=5.0):
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
            "seed": seed,
            "throughput_mbps": throughput,
            "collision_prob": mean_coll,
            "jain_index": jain,
        }
    finally:
        proc.terminate()

if __name__ == "__main__":
    results = []
    for seed in [1, 2, 3]:
        print(f"Running Standard 802.11 for seed {seed}...")
        r = run_standard_80211(seed, port=18100 + seed, sim_time=5.0)
        print(f"Seed {seed}: Throughput = {r['throughput_mbps']:.2f} Mb/s, Coll = {r['collision_prob']:.3f}, Jain = {r['jain_index']:.3f}")
        results.append(r)
        
    tputs = [r['throughput_mbps'] for r in results]
    colls = [r['collision_prob'] for r in results]
    jains = [r['jain_index'] for r in results]
    print("\n=== Standard 802.11 (BEB) Summary ===")
    print(f"Mean Throughput: {np.mean(tputs):.2f} ± {np.std(tputs):.2f} Mb/s")
    print(f"Mean Collision Prob: {np.mean(colls):.3f} ± {np.std(colls):.3f}")
    print(f"Mean Jain Index: {np.mean(jains):.3f} ± {np.std(jains):.3f}")

