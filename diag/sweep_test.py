"""
diag/sweep_test.py
Test static CW execution using direct ns-3 command or ns3gym.
"""
import time, subprocess
from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path
import numpy as np

def run_sim(cw_val, sim_time=5.0, seed=1, dry_run=False, port=15000):
    sim_args = {
        "simTime": sim_time,
        "envStepTime": 0.01,
        "historyLength": 300,
        "agentType": "direct_continuous",
        "scenario": "basic",
        "nWifi": 40,
        "dryRun": dry_run,
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
        for s in range(steps):
            no, rew, dn, inf = env.step(action)
            parts = inf.split("|")
            mb = float(parts[0])
            total_mb += mb
            collision_rates.append(np.mean(no))
            if dn:
                break
        env.close()
        # total_mb is in Megabits (since formula is * 8 / 1024 / 1024)
        throughput = total_mb / sim_time
        mean_coll = np.mean(collision_rates)
        return throughput, mean_coll
    finally:
        proc.terminate()

if __name__ == "__main__":
    t0 = time.time()
    print("Testing CW=255 for 5s...")
    tput, coll = run_sim(255, sim_time=5.0, port=15100)
    print(f"5s result: Throughput = {tput:.2f} Mb/s, Collision = {coll:.3f} (elapsed: {time.time()-t0:.1f}s)")

