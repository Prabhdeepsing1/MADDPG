"""
diag/test_run.py
Quick test to verify running ns3env from python inside myenv
"""
import sys, time
import numpy as np
from ns3gym import ns3env

sim_args = {
    "simTime": 1.0,
    "envStepTime": 0.01,
    "historyLength": 300,
    "agentType": "continuous",
    "scenario": "basic",
    "nWifi": 5,
    "dryRun": True,
}

port = 12345
import subprocess
from ns3gym.start_sim import find_waf_path
waf_pwd = find_waf_path("/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29")

cmd = f'{waf_pwd} --run "linear-mesh'
for k, v in sim_args.items():
    cmd += f" --{k}={v}"
cmd += f' --openGymPort={port}"'

print("Launching:", cmd)
proc = subprocess.Popen(['bash', '-c', cmd])

try:
    env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=0, simArgs=sim_args, debug=False)
    obs = env.reset()
    print("Initial obs shape:", np.array(obs).shape)
    for step in range(10):
        action = [0.0] * 5
        no, rew, dn, inf = env.step(action)
    print("Test passed! 10 steps succeeded. Info:", inf)
    env.close()
finally:
    proc.terminate()

