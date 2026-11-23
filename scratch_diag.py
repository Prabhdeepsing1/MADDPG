import subprocess
import os
import sys
import numpy as np
from ns3gym import ns3env

def run_single(name, agent_type, dry_run, action_fn, n_wifi=5, sim_time=5.0, seed=1, port=18250):
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
    waf_pwd = "/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29/waf"
    cmd = f'{waf_pwd} --run "linear-mesh'
    for k, v in sim_args.items():
        cmd += f" --{k}={v}"
    cmd += f' --openGymPort={port}"'

    print(f"[{name}] Starting ns-3 process on port {port}...")
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        cwd="/home/prabhdeep_singh/workspace/ns-allinone-3.29/ns-3.29",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    try:
        env = ns3env.Ns3Env(port=port, stepTime=0.01, startSim=0, simSeed=seed, simArgs=sim_args, debug=False)
        obs = env.reset()
        steps = int(sim_time / 0.01)
        
        info_mb_sum = 0.0
        step_throughputs = []
        actions_taken = []
        cws_seen = []

        for s in range(steps):
            act = action_fn(s, obs)
            actions_taken.append(act)
            next_obs, rew, done, info = env.step(act)
            
            # parse info: "sentMbytes|CW|activeStations|jain"
            parts = str(info).split("|")
            mb_step = float(parts[0])
            info_mb_sum += mb_step
            step_throughputs.append(mb_step / 0.01)
            if len(parts) > 1:
                cws_seen.append(float(parts[1]))
            obs = next_obs
            if done:
                break
        
        env.close()
        stdout_out, _ = proc.communicate(timeout=10)
        
        # Parse ns-3 stdout line: "Sent mbytes: %.2f\tThroughput: %.3f"
        ns3_sent_mb = None
        ns3_tput = None
        for line in stdout_out.splitlines():
            if "Sent mbytes:" in line:
                print(f"[{name}] Raw ns-3 stdout: {line.strip()}")
                try:
                    p1 = line.split("Sent mbytes:")[1].split("Throughput:")
                    ns3_sent_mb = float(p1[0].strip())
                    ns3_tput = float(p1[1].strip())
                except:
                    pass

        mean_step_tput = float(np.mean(step_throughputs))
        sum_mb_div_time = info_mb_sum / sim_time
        
        print(f"[{name}] info_mb_sum = {info_mb_sum:.4f} Mb (over {len(step_throughputs)} steps)")
        print(f"[{name}] sum_mb_div_time = {sum_mb_div_time:.4f} Mbps")
        print(f"[{name}] mean_step_tput = {mean_step_tput:.4f} Mbps")
        print(f"[{name}] ns3 reported Throughput = {ns3_tput} (Sent Mbytes: {ns3_sent_mb})")
        print(f"[{name}] Mean CW = {np.mean(cws_seen):.2f}")
        return {
            "name": name,
            "ns3_sent_mb": ns3_sent_mb,
            "ns3_tput": ns3_tput,
            "info_mb_sum": info_mb_sum,
            "mean_step_tput": mean_step_tput,
            "mean_cw": float(np.mean(cws_seen)) if cws_seen else 0.0,
        }
    except Exception as e:
        proc.terminate()
        raise e

if __name__ == "__main__":
    print("=== TEST 1: BEB (dryRun=1) N=5, simTime=5.0s, seed=1 ===")
    run_single("BEB", "continuous", dry_run=True, action_fn=lambda s, o: [0.0]*5, n_wifi=5, sim_time=5.0, seed=1, port=18301)
