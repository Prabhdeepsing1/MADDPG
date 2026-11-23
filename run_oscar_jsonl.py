#!/usr/bin/env python3
"""Step 4 Task 3 — OSCAR DDPG action diagnostics with JSONL logging.

Runs the OSCAR single-agent DDPG (seed 1) and logs per-step per-station
action so it can be compared with MADDPG variants on the same axes:
mean action a per epoch and std across stations.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import wandb

from exceptions import AlreadyRunningException
from wrappers import EnvWrapper
from agents.our_ddpg.Our_DDPG import DDPG as OscarDDPG
from agents.our_ddpg.utils import ReplayBuffer as OscarReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--simTime", type=float, default=15)
    ap.add_argument("--nWifi", type=int, default=5)
    ap.add_argument("--jsonl", default="oscar_seed1.jsonl")
    ap.add_argument("--summary_json", default="oscar_seed1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N = args.nWifi
    step_time = 0.01
    steps_per_ep = int(args.simTime / step_time)
    start_timesteps = 300
    expl_noise = 0.5
    expl_noise_min = 0.05
    expl_noise_decay = 0.99
    current_expl_noise = expl_noise
    batch_size = 32
    discount = 0.7  # OSCAR default

    sim_args = {
        "simTime": args.simTime,
        "envStepTime": step_time,
        "historyLength": 300,
        "agentType": "continuous",
        "scenario": "basic",
        "nWifi": N,
        "dryRun": False,
        "seed": args.seed,
    }

    run = wandb.init(
        name=f"{N} OSCAR DDPG",
        entity="1602-23-733-162-vasavi-college-of-engineering",
        project="contention_window",
        tags=[f"{args.episodes}ep training", f"{args.simTime}s",
              f"{N} nWifi", "OSCAR", "single-agent DDPG"],
        reinit=True,
        config={
            "nWifi": N,
            "simTime": args.simTime,
            "stepTime": step_time,
            "episodes": args.episodes,
            "seed": args.seed,
            "start_timesteps": start_timesteps,
            "batch_size": batch_size,
            "discount": discount,
            "tau": 1e-3,
            "expl_noise": expl_noise,
            "expl_noise_min": expl_noise_min,
            "expl_noise_decay": expl_noise_decay,
            "method": "oscar_ddpg",
        },
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    env = EnvWrapper(1, **sim_args)
    policy = OscarDDPG(state_dim=1, action_dim=1, max_action=1.0,
                       discount=discount, tau=1e-3)
    replay_buffer = OscarReplayBuffer(1, 1)

    time_step = 0
    ep_throughputs = []
    ep_actions = []
    step_actions = []
    step_obs = []
    step_tputs = []

    f = open(args.jsonl, "w")

    for ep in range(args.episodes):
        try:
            env.run()
        except AlreadyRunningException:
            pass

        obs_list = env.reset()
        state = obs_list[0][0][:1]  # shape (1,)
        ep_actions = []
        ep_tputs = []

        for step in range(steps_per_ep):
            if time_step < start_timesteps:
                real_action = np.random.uniform(0.0, 6.0)
                norm_action = (real_action / 3.0) - 1.0
            else:
                a_pred = policy.select_action(state)
                noisy_a = np.clip(a_pred + np.random.normal(0, current_expl_noise, size=1), -1.0, 1.0)
                norm_action = noisy_a[0]
                real_action = 3.0 * (norm_action + 1.0)  # [0, 6]

            # cw.cc: cw = 2^(action + 4); action passed to env is real_action
            # OSCAR logs a_cont = real_action + 4.0 in [4, 10]
            a_cont = real_action + 4.0
            # Same action to all stations (single agent)
            joint_action = np.full((N, 1), a_cont, dtype=np.float32)
            ep_actions.append(a_cont)

            env_action = [[real_action] * N]
            next_obs_list, rewards, dones, infos = env.step(env_action)

            next_state = next_obs_list[0][0][:1]
            rew = rewards[0]
            done = dones[0]
            info_str = infos[0]

            sent_mbits = float(str(info_str).split("|")[0])
            tput_mbps = sent_mbits / step_time
            ep_tputs.append(tput_mbps)

            replay_buffer.add(state, np.array([norm_action]), next_state, rew, done)
            state = next_state

            if time_step >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            f.write(json.dumps({
                "episode": ep,
                "step": time_step,
                "obs": next_obs_list[0].flatten().tolist(),
                "joint_action": joint_action.flatten().tolist(),
                "throughput_mbps": tput_mbps,
                "reward": float(rew),
                "done": bool(done),
            }) + "\n")

            wandb.log({
                "step": time_step,
                "episode": ep,
                "global_reward": float(rew),
                "instantaneous_throughput_mbps": tput_mbps,
                "frame_loss_rate": float(np.mean(next_obs_list[0])),
                "mean_action": float(np.mean(joint_action)),
            })

            time_step += 1
            if done:
                break

        current_expl_noise = max(expl_noise_min, current_expl_noise * expl_noise_decay)
        mean_ep_tput = float(np.mean(ep_tputs)) if ep_tputs else 0.0
        ep_throughputs.append(mean_ep_tput)
        step_actions.append(float(np.mean(ep_actions)))
        wandb.log({
            "step": time_step,
            "episode": ep,
            "mean_episode_throughput_mbps": mean_ep_tput,
            "episode_total_reward": float(np.sum(ep_tputs)),
            "episode_mean_action": float(np.mean(ep_actions)),
        })
        print(f"OSCAR Ep {ep} | mean a={np.mean(ep_actions):.3f} | Tput={mean_ep_tput:.2f} Mb/s")

        env.close()
        if ep + 1 < args.episodes:
            env = EnvWrapper(1, **sim_args)

    f.close()
    env.close()

    with open(args.summary_json, "w") as sf:
        json.dump({
            "config": {"seed": args.seed, "episodes": args.episodes,
                       "simTime": args.simTime, "nWifi": N, "method": "oscar_ddpg"},
            "episodes": [{"episode": e, "mean_throughput_mbps": t, "mean_action": a}
                         for e, t, a in zip(range(len(ep_throughputs)), ep_throughputs, step_actions)],
            "final_mean_throughput_mbps": ep_throughputs[-1] if ep_throughputs else 0.0,
        }, sf, indent=2)
    print("Saved", args.summary_json)
    run.finish()


if __name__ == "__main__":
    main()