#!/usr/bin/env python3
"""Step 4 Task 5 — Throughput formula verification.

For one seed each of BEB (dryRun) and current MADDPG-style action input:
  * reads the raw byte counter delta (info[0] sentMbytes per step)
  * computes the Python-side throughput formula used by maddpg_train.py
  * prints the raw counters, reward, and logged Mbps side by side
  * checks the reward back-conversion reward = D_mb / (5*150*dt) * 10
    (C++ hard-codes 5 regardless of N)
"""
import numpy as np
from ns3gym import ns3env

WAF_PWD = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
NS3_DIR = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29"


def run_check(n_wifi, sim_time=5.0, seed=1, port=19500, dry_run=True, action_a=4):
    print("=" * 70)
    print(f"N={n_wifi} dryRun={dry_run} seed={seed} a={action_a}")
    print("=" * 70)
    sim_args = {
        "simTime": sim_time, "envStepTime": 0.01, "historyLength": 300,
        "agentType": "continuous", "scenario": "basic", "nWifi": n_wifi,
        "dryRun": 1 if dry_run else 0, "seed": seed,
    }
    cmd = f'{WAF_PWD} --run "linear-mesh'
    for k, v in sim_args.items():
        cmd += f" --{k}={v}"
    cmd += f' --openGymPort={port}"'

    proc = subprocess.Popen(['bash', '-c', cmd], cwd=NS3_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=seed,
                        simArgs=sim_args, debug=False)
    env.reset()
    steps = int(sim_time / 0.01)
    env_act = [action_a - 4.0] * n_wifi
    info_mb = []
    rewards = []
    for s in range(steps):
        next_obs, rew, done, info = env.step(env_act)
        parts = str(info).split("|")
        info_mb.append(float(parts[0]))          # sent Mbits this step (all stations)
        rewards.append(float(rew))
        if done:
            break
    env.close()
    try:
        proc.communicate(timeout=10)
    except Exception:
        proc.terminate()

    info_mb = np.array(info_mb)
    rewards = np.array(rewards)
    tputs = info_mb / 0.01                       # logged Mbps (Python formula)

    # raw byte counters: info[0] is Mbits per step => bytes = Mbits*1e6/8
    raw_bytes_step = info_mb * 1e6 / 8
    total_bytes = raw_bytes_step.sum()

    print(f"  steps logged                = {len(info_mb)}")
    print(f"  raw bytes total (from info) = {total_bytes:.0f}")
    print(f"  mean info[0] per step (Mb)  = {info_mb.mean():.6f}")
    print(f"  logged Mbps = info/dt       = {tputs.mean():.4f}")
    # reward back-conversion: reward = D_mb/(5*150*dt)*10  => D_mb = r*0.075
    r = rewards[rewards > 0].mean()
    back_mb = r * (5.0 * 150.0 * 0.01) / 10.0     # = D_mb per step
    back_mbps = back_mb / 0.01
    print(f"  mean reward (r>0)           = {r:.6f}")
    print(f"  reward->D_mb = r*0.075      = {back_mb:.6f} Mb/step")
    print(f"  reward->Mbps = D_mb/dt      = {back_mbps:.4f}")
    print(f"  ratio logged/reward         = {tputs[rewards > 0].mean() / r:.4f}  (expect 75)")
    print(f"  consistency check           = "
          f"{np.allclose(info_mb[rewards > 0], back_mb, rtol=1e-3)}")
    return {"n": n_wifi, "dry_run": dry_run, "mean_tput_mbps": float(tputs.mean()),
            "mean_reward": float(r), "total_bytes": float(total_bytes)}


import subprocess  # noqa: E402

if __name__ == "__main__":
    out = []
    out.append(run_check(5, port=19501, dry_run=True, action_a=4))   # BEB N=5
    out.append(run_check(10, port=19502, dry_run=True, action_a=4))  # BEB N=10
    out.append(run_check(5, port=19503, dry_run=False, action_a=7))  # MADDPG-style N=5
    out.append(run_check(10, port=19504, dry_run=False, action_a=7)) # MADDPG-style N=10
    import json
    with open("throughput_formula_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved throughput_formula_check.json")