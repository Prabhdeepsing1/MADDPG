"""
run_action_diagnostics.py
Tasks 1 & 3:
1. Verify throughput metric: compare raw byte counters and logged throughput for BEB, OSCAR DDPG, and MADDPG.
3. Action diagnostics: For OSCAR DDPG and MADDPG (seed 1, 4 episodes, 5s), log per-epoch mean action a and CW.
   Generate SVG comparison plot and analyze whether MADDPG saturates, oscillates, or converges.
"""
import copy
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from agents.our_ddpg.Our_DDPG import DDPG as OscarDDPG
from agents.our_ddpg.utils import ReplayBuffer as OscarReplayBuffer
from agents.maddpg.model import Actor as MaddpgActor, MeanFieldCritic
from agents.maddpg.utils import MADDPGReplayBuffer
from wrappers import EnvWrapper

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_oscar_ddpg(seed=1, episodes=4, sim_time=5.0, n_wifi=5):
    print("\n========================================================")
    print(f"RUNNING OSCAR DDPG (Seed {seed}, {episodes} eps, {sim_time}s)")
    print("========================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)

    step_time = 0.01
    steps_per_ep = int(sim_time / step_time)
    start_timesteps = 100
    expl_noise = 0.1
    batch_size = 32

    sim_args = {
        "simTime": sim_time,
        "envStepTime": step_time,
        "historyLength": 300,
        "agentType": "continuous",
        "scenario": "basic",
        "nWifi": n_wifi,
        "dryRun": False,
        "seed": seed,
    }

    env = EnvWrapper(1, **sim_args)
    policy = OscarDDPG(state_dim=1, action_dim=1, max_action=1.0, discount=0.7, tau=1e-3)
    replay_buffer = OscarReplayBuffer(1, 1)

    time_step = 0
    ep_results = []
    step_actions_all = []
    step_tputs_all = []

    for ep in range(episodes):
        try:
            env.run()
        except Exception:
            pass
        obs_list = env.reset()
        state = obs_list[0][0][:1]  # shape (1,)
        ep_actions = []
        ep_cws = []
        ep_tputs = []
        ep_reward = 0.0

        for step in range(steps_per_ep):
            if time_step < start_timesteps:
                # Random in [0, 6]
                real_action = np.random.uniform(0.0, 6.0)
                norm_action = (real_action / 3.0) - 1.0  # in [-1, 1]
            else:
                a_pred = policy.select_action(state)
                noisy_a = np.clip(a_pred + np.random.normal(0, expl_noise, size=1), -1.0, 1.0)
                norm_action = noisy_a[0]
                real_action = 3.0 * (norm_action + 1.0)  # [0, 6]

            # In continuous mode, cw.cc computes cw_val = pow(2, action + 4)
            # So continuous action a in [4, 10] is real_action + 4.0
            a_cont = real_action + 4.0
            cw_val = 2.0 ** a_cont
            ep_actions.append(a_cont)
            ep_cws.append(cw_val)
            step_actions_all.append(a_cont)

            # Env expects action per station
            env_action = [[real_action] * n_wifi]
            next_obs_list, rewards, dones, infos = env.step(env_action)

            next_state = next_obs_list[0][0][:1]
            rew = rewards[0]
            done = dones[0]
            info_str = infos[0]

            # Throughput
            sent_mbits = float(str(info_str).split("|")[0])
            tput_mbps = sent_mbits / step_time
            ep_tputs.append(tput_mbps)
            step_tputs_all.append(tput_mbps)
            ep_reward += rew

            # Buffer
            replay_buffer.add(state, np.array([norm_action]), next_state, rew, done)
            state = next_state

            if time_step >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            time_step += 1
            if done:
                break

        ep_res = {
            "episode": ep,
            "mean_a": round(float(np.mean(ep_actions)), 3),
            "mean_cw": round(float(np.mean(ep_cws)), 1),
            "throughput_mbps": round(float(np.mean(ep_tputs)), 2),
            "reward": round(float(ep_reward), 2),
        }
        ep_results.append(ep_res)
        print(f"OSCAR Ep {ep} -> mean action a={ep_res['mean_a']:.3f}, CW={ep_res['mean_cw']:.1f}, Tput={ep_res['throughput_mbps']:.2f} Mbps, Reward={ep_res['reward']:.2f}")

    env.close()

    return {
        "method": "OSCAR DDPG",
        "episodes": ep_results,
        "step_actions": step_actions_all,
        "step_tputs": step_tputs_all,
    }

def run_maddpg_diag(seed=1, episodes=4, sim_time=5.0, n_wifi=5):
    print("\n========================================================")
    print(f"RUNNING MADDPG (Seed {seed}, {episodes} eps, {sim_time}s)")
    print("========================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)

    step_time = 0.01
    steps_per_ep = int(sim_time / step_time)
    start_timesteps = 100
    expl_noise = 0.5
    expl_noise_min = 0.05
    expl_noise_decay = 0.99
    current_expl_noise = expl_noise
    batch_size = 32
    discount = 0.99
    tau = 0.005

    sim_args = {
        "simTime": sim_time,
        "envStepTime": step_time,
        "historyLength": 300,
        "agentType": "continuous",
        "scenario": "basic",
        "nWifi": n_wifi,
        "dryRun": False,
        "seed": seed,
    }

    env = EnvWrapper(1, **sim_args)

    local_actor = MaddpgActor(obs_dim=1, act_dim=1, seed=seed).to(DEVICE)
    target_actor = copy.deepcopy(local_actor)
    local_critic = MeanFieldCritic().to(DEVICE)
    target_critic = copy.deepcopy(local_critic)

    actor_opt = torch.optim.Adam(local_actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(local_critic.parameters(), lr=1e-3)
    replay_buffer = MADDPGReplayBuffer(n_wifi, max_size=10000)

    time_step = 0
    ep_results = []
    step_actions_all = []
    step_tputs_all = []

    def soft_update(tgt, src, tau):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    for ep in range(episodes):
        try:
            env.run()
        except Exception:
            pass
        obs_list = env.reset()
        obs = obs_list[0]  # shape (N, 1)
        ep_actions = []
        ep_cws = []
        ep_tputs = []
        ep_reward = 0.0

        for step in range(steps_per_ep):
            if time_step < start_timesteps:
                joint_action = np.random.uniform(4.0, 10.0, size=(n_wifi, 1)).astype(np.float32)
            else:
                local_actor.eval()
                with torch.no_grad():
                    o_t = torch.FloatTensor(obs).to(DEVICE)
                    a_t = local_actor(o_t).cpu().numpy()
                local_actor.train()
                a_t += np.random.normal(0, current_expl_noise, size=(n_wifi, 1))
                joint_action = np.clip(a_t, 4.0, 10.0).astype(np.float32)

            mean_a_step = float(np.mean(joint_action))
            mean_cw_step = float(np.mean(2.0 ** joint_action))
            ep_actions.append(mean_a_step)
            ep_cws.append(mean_cw_step)
            step_actions_all.append(mean_a_step)

            actions_for_env = [joint_action - 4.0]
            next_obs_list, rewards, dones, infos = env.step(actions_for_env)

            next_obs = next_obs_list[0]
            reward = rewards[0]
            done = dones[0]

            alpha = 0.5
            local_success = 1.0 - obs.flatten()
            mean_success = np.mean(local_success) + 1e-6
            norm_local_success = local_success / mean_success
            reward_vec = alpha * reward + (1.0 - alpha) * reward * norm_local_success

            replay_buffer.add(obs, joint_action, reward_vec, next_obs, done)
            obs = next_obs
            ep_reward += reward

            # Training
            if time_step >= start_timesteps and len(replay_buffer) > batch_size:
                jo, ja, r, jno, nd = replay_buffer.sample(batch_size)
                jo, ja, r, jno, nd = jo.to(DEVICE), ja.to(DEVICE), r.to(DEVICE), jno.to(DEVICE), nd.to(DEVICE)

                with torch.no_grad():
                    jno_reshaped = jno.view(-1, 1)
                    a_tgt = target_actor(jno_reshaped).view(-1, n_wifi)
                    sum_jno = jno.sum(dim=1, keepdim=True)
                    sum_ja_tgt = a_tgt.sum(dim=1, keepdim=True)
                    mean_jno = (sum_jno - jno) / (n_wifi - 1)
                    mean_ja_tgt = (sum_ja_tgt - a_tgt) / (n_wifi - 1)

                    target_q = target_critic(jno.view(-1, 1), a_tgt.view(-1, 1), mean_jno.view(-1, 1), mean_ja_tgt.view(-1, 1))
                    y = r.view(-1, 1) + discount * target_q * nd.repeat(1, n_wifi).view(-1, 1)

                sum_jo = jo.sum(dim=1, keepdim=True)
                sum_ja = ja.sum(dim=1, keepdim=True)
                mean_jo = (sum_jo - jo) / (n_wifi - 1)
                mean_ja = (sum_ja - ja) / (n_wifi - 1)
                cur_q = local_critic(jo.view(-1, 1), ja.view(-1, 1), mean_jo.view(-1, 1), mean_ja.view(-1, 1))

                critic_loss = F.mse_loss(cur_q, y)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(local_critic.parameters(), 1.0)
                critic_opt.step()

                # Actor
                actor_opt.zero_grad()
                a_pred = local_actor(jo.view(-1, 1)).view(-1, n_wifi)
                sum_new_ja = a_pred.sum(dim=1, keepdim=True)
                mean_new_ja = (sum_new_ja - a_pred) / (n_wifi - 1)
                actor_q = local_critic(jo.view(-1, 1), a_pred.view(-1, 1), mean_jo.view(-1, 1), mean_new_ja.view(-1, 1))
                actor_loss = -actor_q.mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(local_actor.parameters(), 1.0)
                actor_opt.step()

                soft_update(target_actor, local_actor, tau)
                soft_update(target_critic, local_critic, tau)

            time_step += 1

            sent_mbits = float(str(infos[0]).split("|")[0])
            tput_mbps = sent_mbits / step_time
            ep_tputs.append(tput_mbps)
            step_tputs_all.append(tput_mbps)

            if done:
                break

        current_expl_noise = max(expl_noise_min, current_expl_noise * expl_noise_decay)
        ep_res = {
            "episode": ep,
            "mean_a": round(float(np.mean(ep_actions)), 3),
            "mean_cw": round(float(np.mean(ep_cws)), 1),
            "throughput_mbps": round(float(np.mean(ep_tputs)), 2),
            "reward": round(float(ep_reward), 2),
        }
        ep_results.append(ep_res)
        print(f"MADDPG Ep {ep} -> mean action a={ep_res['mean_a']:.3f}, CW={ep_res['mean_cw']:.1f}, Tput={ep_res['throughput_mbps']:.2f} Mbps, Reward={ep_res['reward']:.2f}")

    env.close()

    return {
        "method": "MADDPG",
        "episodes": ep_results,
        "step_actions": step_actions_all,
        "step_tputs": step_tputs_all,
    }

def generate_svg_plot(oscar_data, maddpg_data, out_path="action_diagnostics.svg"):
    # Generate an SVG showing action trajectory a(t) for OSCAR vs MADDPG
    oscar_a = oscar_data["step_actions"]
    maddpg_a = maddpg_data["step_actions"]
    n_pts = min(len(oscar_a), len(maddpg_a))

    # Downsample for SVG readability (window of 10)
    w = 10
    oscar_smooth = [float(np.mean(oscar_a[i:i+w])) for i in range(0, n_pts, w)]
    maddpg_smooth = [float(np.mean(maddpg_a[i:i+w])) for i in range(0, n_pts, w)]
    x_steps = [i for i in range(0, n_pts, w)]

    width, height = 800, 450
    margin = 60

    min_x, max_x = 0, n_pts
    min_y, max_y = 3.5, 10.5

    def to_svg_coords(x, y):
        sx = margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)
        sy = (height - margin) - (y - min_y) / (max_y - min_y) * (height - 2 * margin)
        return sx, sy

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#1a1b26"/>',
        f'<text x="{width//2}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" fill="#c0caf5" text-anchor="middle">Contention Window Action Exponent a(t) Over Time [4.0 = CW:16, 10.0 = CW:1024]</text>',
        # Axes
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#414868" stroke-width="2"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#414868" stroke-width="2"/>',
    ]

    # Y-ticks
    for y_val in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        sx, sy = to_svg_coords(0, y_val)
        cw = int(2**y_val)
        svg_lines.append(f'<line x1="{margin-5}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#24283b" stroke-width="1" stroke-dasharray="4"/>')
        svg_lines.append(f'<text x="{margin-10}" y="{sy+4}" font-family="sans-serif" font-size="11" fill="#7aa2f7" text-anchor="end">{y_val:.0f} (CW:{cw})</text>')

    # X-ticks (episodes)
    steps_per_ep = 500
    for ep in range(5):
        step_val = ep * steps_per_ep
        sx, sy = to_svg_coords(step_val, 4.0)
        svg_lines.append(f'<line x1="{sx}" y1="{height-margin}" x2="{sx}" y2="{height-margin+5}" stroke="#414868" stroke-width="2"/>')
        svg_lines.append(f'<text x="{sx}" y="{height-margin+20}" font-family="sans-serif" font-size="11" fill="#7aa2f7" text-anchor="middle">Ep {ep}</text>')

    # Plot OSCAR DDPG line (coral / orange #ff9e64)
    pts_oscar = []
    for x, y in zip(x_steps, oscar_smooth):
        sx, sy = to_svg_coords(x, y)
        pts_oscar.append(f"{sx:.1f},{sy:.1f}")
    svg_lines.append(f'<polyline points="{" ".join(pts_oscar)}" fill="none" stroke="#ff9e64" stroke-width="2.5"/>')

    # Plot MADDPG line (cyan #7dcfff)
    pts_maddpg = []
    for x, y in zip(x_steps, maddpg_smooth):
        sx, sy = to_svg_coords(x, y)
        pts_maddpg.append(f"{sx:.1f},{sy:.1f}")
    svg_lines.append(f'<polyline points="{" ".join(pts_maddpg)}" fill="none" stroke="#7dcfff" stroke-width="2.5"/>')

    # Legend
    svg_lines.extend([
        f'<rect x="{width-240}" y="45" width="20" height="4" fill="#ff9e64"/>',
        f'<text x="{width-210}" y="50" font-family="sans-serif" font-size="12" fill="#c0caf5">OSCAR DDPG (Single Agent)</text>',
        f'<rect x="{width-240}" y="65" width="20" height="4" fill="#7dcfff"/>',
        f'<text x="{width-210}" y="70" font-family="sans-serif" font-size="12" fill="#c0caf5">MADDPG (Multi-Agent CTDE)</text>',
        '</svg>'
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(svg_lines))
    print(f"Saved SVG plot to {out_path}")

def main():
    oscar_res = run_oscar_ddpg(seed=1, episodes=4, sim_time=5.0, n_wifi=5)
    time.sleep(2.0)
    maddpg_res = run_maddpg_diag(seed=1, episodes=4, sim_time=5.0, n_wifi=5)

    summary = {
        "oscar": oscar_res["episodes"],
        "maddpg": maddpg_res["episodes"],
    }
    with open("action_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)

    generate_svg_plot(oscar_res, maddpg_res, "action_diagnostics.svg")
    print("\nAction Diagnostics Complete!")

if __name__ == "__main__":
    main()
