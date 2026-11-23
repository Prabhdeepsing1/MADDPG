#!/usr/bin/env python3
"""Step 4 Task 4 — Fixed-CW sweep (constant a = 4..10), 3 seeds.

Holds the CW exponent a constant for all stations for the whole episode and
records throughput, so we can find the achievable ceiling per a.

In continuous mode cw.cc does cw_val = 2^(action + 4) where the env action is
a-4, so sending [a-4]*N gives CW = 2^a for every station.
"""
import json
import os
import subprocess
import numpy as np
from ns3gym import ns3env

WAF_PWD = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
NS3_DIR = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29"


def run_one(a, n_wifi, sim_time, seed, port):
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": "continuous",
        "scenario": "basic",
        "nWifi": n_wifi,
        "dryRun": 0,
        "seed": seed,
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
    env_act = [a - 4.0] * n_wifi   # cw.cc: cw = 2^(action+4) = 2^a
    step_tputs, cws, per_sta_obs = [], [], []
    for s in range(steps):
        next_obs, rew, done, info = env.step(env_act)
        parts = str(info).split("|")
        step_tputs.append(float(parts[0]) / 0.01)
        if len(parts) > 1:
            cws.append(float(parts[1]))
        per_sta_obs.append(np.array(next_obs, dtype=float).flatten())
        if done:
            break
    env.close()
    try:
        proc.communicate(timeout=10)
    except Exception:
        proc.terminate()

    mean_tput = float(np.mean(step_tputs))
    jain = 1.0
    po = np.array(per_sta_obs)
    if po.size:
        # success proxy = 1 - loss rate, per station
        w = np.maximum(1.0 - po.mean(axis=0), 1e-6)
        jain = float(w.sum() ** 2 / (n_wifi * (w ** 2).sum()))
    return {
        "a": a, "cw": int(2 ** a), "seed": seed,
        "mean_tput_mbps": round(mean_tput, 3),
        "mean_cw": round(float(np.mean(cws)), 2) if cws else None,
        "jain": round(jain, 4),
    }


def main():
    actions = [4, 5, 6, 7, 8, 9, 10]
    seeds = [1, 2, 3]
    n_wifi, sim_time = 5, 5.0
    port = 19600
    out = {}
    for a in actions:
        rows = []
        for s in seeds:
            r = run_one(a, n_wifi, sim_time, s, port)
            port += 1
            rows.append(r)
            print(f"  a={a} (CW={2**a}) seed={s}: tput={r['mean_tput_mbps']:.2f} Mbps, "
                  f"CW_mean={r['mean_cw']}, jain={r['jain']:.4f}")
        ts = [r["mean_tput_mbps"] for r in rows]
        out[a] = rows
        print(f"  >>> a={a} (CW={2**a}): mean {np.mean(ts):.2f} ± {np.std(ts):.2f} Mbps")
    with open("fixed_cw_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved fixed_cw_sweep.json")


if __name__ == "__main__":
    main()