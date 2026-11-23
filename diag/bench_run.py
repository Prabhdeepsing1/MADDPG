"""
diag/bench_run.py
Measure runtime of a 15s simulation with N=40 stations.
"""
import time, os, subprocess
import numpy as np
from ns3gym import ns3env

sim_args = {
    "simTime": 15.0,
    "envStepTime": 0.01,
    "historyLength": 300,
    "agentType": "continuous",
    "scenario": "basic",
    "nWifi": 40,
    "dryRun": True,
    "seed": 1,
}

port = 14500
waf_pwd = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"

cmd = f'{waf_pwd} --run "linear-mesh'
for k, v in sim_args.items():
    cmd += f" --{k}={v}"
cmd += f' --openGymPort={port}"'

print("Launching:", cmd)
t0 = time.time()
proc = subprocess.Popen(['bash', '-c', cmd], cwd="/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29")

try:
    env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=1, simArgs=sim_args, debug=False)
    obs = env.reset()
    steps = int(15.0 / 0.01)
    print(f"Connected in {time.time()-t0:.2f}s. Stepping {steps} steps...")
    
    total_rx_mb = 0.0
    action = [0.0] * 40
    
    for s in range(steps):
        no, rew, dn, inf = env.step(action)
        if s % 300 == 0:
            print(f"Step {s}/{steps} ({time.time()-t0:.1f}s) - info: {inf}")
        if dn:
            break
            
    print(f"Finished {steps} steps in {time.time()-t0:.2f}s. Final info: {inf}")
    env.close()
finally:
    proc.terminate()

