import numpy as np
import torch
import torch.nn.functional as F
import argparse
import json
import os
import time
import wandb
import tqdm
import copy
from collections import deque

from exceptions import AlreadyRunningException
from wrappers import EnvWrapper

from agents.maddpg.model import MeanFieldCritic, Actor, CentralizedCritic, create_homogeneous_actors
from agents.maddpg.utils import MADDPGReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nWifi", default=5, type=int, help="Number of WiFi stations (N)")
    parser.add_argument("--scenario", default="basic")
    parser.add_argument("--simTime", default=15, type=float)
    parser.add_argument("--stepTime", default=0.01, type=float)
    parser.add_argument("--history_length", default=300, type=int)
    parser.add_argument("--seed", default=1, type=int,
                        help="Positive random seed (ns-3 rejects zero)")
    parser.add_argument("--start_timesteps", default=300, type=int)
    parser.add_argument("--max_timesteps", default=1000000, type=int)
    parser.add_argument("--expl_noise", default=0.5, type=float)
    parser.add_argument("--expl_noise_min", default=0.05, type=float)
    parser.add_argument("--expl_noise_decay", default=0.99, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)
    parser.add_argument("--episodes", default=12, type=int)
    parser.add_argument("--buffer_size", default=100000, type=int,
                        help="Replay buffer max size")
    parser.add_argument("--logit_noise", action="store_true",
                        help="Add exploration noise in logit space (pre-sigmoid)")
    parser.add_argument("--use_centralized_critic", action="store_true",
                        help="Use CentralizedCritic + N separate actors (true MADDPG) instead of shared actor + MeanFieldCritic")
    parser.add_argument("--jsonl", default="", type=str,
                        help="If set, append per-step obs/joint-action/throughput to this JSONL file")
    parser.add_argument("--summary_json", default="", type=str,
                        help="If set, write final per-episode summary + config to this JSON file")
    parser.add_argument("--ceiling_mbps", default=37.24, type=float,
                        help="Throughput ceiling used for relative convergence threshold")
    parser.add_argument("--convergence_frac", default=0.98, type=float,
                        help="Fraction of ceiling that marks convergence")
    args = parser.parse_args()
    if args.seed <= 0:
        parser.error("--seed must be a positive integer because ns-3 rejects seed 0")

    # 1. Initialization
    N = args.nWifi
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Initializing MADDPG with {N} agents...")
    convergence_threshold = args.convergence_frac * args.ceiling_mbps
    print(
        f"Convergence threshold: {convergence_threshold:.5f} Mb/s "
        f"({args.convergence_frac:.4f} x {args.ceiling_mbps:.5f} Mb/s)"
    )

    if args.use_centralized_critic:
        # True MADDPG: N separate actors + 1 CentralizedCritic
        print("  Using CentralizedCritic + N separate actors (true MADDPG)")
        local_actors = create_homogeneous_actors(N, obs_dim=1, act_dim=1, seed=args.seed)
        target_actors = [copy.deepcopy(a) for a in local_actors]
        actor_optimizers = [torch.optim.Adam(a.parameters(), lr=args.lr_actor) for a in local_actors]

        local_critic = CentralizedCritic(n_agents=N, obs_dim=1, act_dim=1, seed=args.seed).to(device)
        target_critic = copy.deepcopy(local_critic)
        critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=args.lr_critic)
    else:
        # Legacy: Shared actor + MeanFieldCritic
        print("  Using shared actor + MeanFieldCritic (legacy)")
        local_actor = Actor(obs_dim=1, act_dim=1, seed=args.seed).to(device)
        target_actor = copy.deepcopy(local_actor)
        local_critic = MeanFieldCritic().to(device)
        target_critic = copy.deepcopy(local_critic)
        actor_optimizer = torch.optim.Adam(local_actor.parameters(), lr=args.lr_actor)
        critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=args.lr_critic)

    # Replay Buffer
    replay_buffer = MADDPGReplayBuffer(N, max_size=args.buffer_size)

    # Env parameters
    sim_args = {
        "simTime": args.simTime,
        "envStepTime": args.stepTime,
        "historyLength": args.history_length,
        "agentType": "continuous",
        "scenario": args.scenario,
        "nWifi": N,
        "seed": args.seed,
        "rng": args.seed,
    }

    wtags = [f"{args.episodes}ep training", f"{args.simTime}s", f"{N} nWifi", f"envStep {args.stepTime}", "MADDPG", "CTDE"]
    run = wandb.init(
        name=f"{N} MADDPG CTDE",
        entity="1602-23-733-162-vasavi-college-of-engineering",
        project="contention_window",
        tags=wtags,
        reinit=True,
        config=vars(args),
    )

    # ------------------------------------------------------------------
    # Metric axis alignment: every metric will be plotted vs. "step"
    # ------------------------------------------------------------------
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")


    threads_no = 1
    env = EnvWrapper(threads_no, **sim_args)

    steps_per_ep = int(args.simTime / args.stepTime)
    time_step = 0
    current_expl_noise = args.expl_noise

    # ------------------------------------------------------------------
    # Convergence tracker state (initialised once, persists across eps)
    # ------------------------------------------------------------------
    _throughput_window: deque = deque(maxlen=100)   # rolling avg window
    _convergence_logged: bool  = False               # fire-once flag
    _all_episode_throughputs: list = []              # per-episode mean throughput
    _all_episode_actions: list = []                  # per-episode mean action
    _all_episode_fairness: list = []
    _all_episode_fairness_rolling: list = []
    _all_episode_fairness_derived: list = []
    _all_episode_rx_packets: list = []
    _convergence_epoch = None

    for episode in range(args.episodes):
        try:
            env.run()
        except AlreadyRunningException:
            pass

        # Reset env
        obs_list = env.reset() # shape (threads_no, N, 1)
        obs = obs_list[0]      # shape (N, 1)
        
        episode_reward = 0
        episode_throughputs: list = []  # accumulate per-step throughput for ep mean
        episode_actions: list = []      # per-step joint_action for this episode
        episode_fairness: list = []
        episode_fairness_episode: list = []
        episode_fairness_derived: list = []
        episode_rx_packets: list = []
        episode_rx_packets_final: list = []

        with tqdm.trange(1, steps_per_ep + 1) as t:
            for step in t:
                # 2. Rollout / Action Selection
                joint_action = np.zeros((N, 1), dtype=np.float32)

                if time_step < args.start_timesteps:
                    # Random exploration initially
                    joint_action = np.random.uniform(4.0, 10.0, size=(N, 1)).astype(np.float32)
                else:
                    if args.use_centralized_critic:
                        # N separate actors
                        with torch.no_grad():
                            for i in range(N):
                                local_actors[i].eval()
                            o_tensor = torch.FloatTensor(obs).to(device)  # shape (N, 1)
                            actions_list = []
                            for i in range(N):
                                if args.logit_noise:
                                    # Logit-space noise
                                    x = F.relu(local_actors[i].fc1(o_tensor[i:i+1]))
                                    x = F.relu(local_actors[i].fc2(x))
                                    logit = local_actors[i].fc3(x)
                                    logit_noisy = logit + torch.randn_like(logit) * current_expl_noise
                                    a_i = (local_actors[i].ACTION_LOW +
                                           (local_actors[i].ACTION_HIGH - local_actors[i].ACTION_LOW)
                                           * torch.sigmoid(logit_noisy))
                                else:
                                    a_i = local_actors[i](o_tensor[i:i+1])
                                actions_list.append(a_i)
                            a_tensor = torch.cat(actions_list, dim=0).cpu().numpy()  # (N, 1)
                            if not args.logit_noise:
                                a_tensor += np.random.normal(
                                    0, current_expl_noise, size=(N, 1)
                                )
                            joint_action = np.clip(a_tensor, 4.0, 10.0).astype(np.float32)
                            for i in range(N):
                                local_actors[i].train()
                    else:
                        # Shared actor (legacy)
                        local_actor.eval()
                        with torch.no_grad():
                            o_tensor = torch.FloatTensor(obs).to(device)  # shape (N, 1)
                            if args.logit_noise:
                                logit = local_actor.fc3(torch.relu(local_actor.fc2(
                                            torch.relu(local_actor.fc1(o_tensor)))))
                                logit_noisy = logit + torch.randn_like(logit) * current_expl_noise
                                a_tensor = (local_actor.ACTION_LOW +
                                            (local_actor.ACTION_HIGH - local_actor.ACTION_LOW)
                                            * torch.sigmoid(logit_noisy)).cpu().numpy()
                                joint_action = np.clip(a_tensor, 4.0, 10.0).astype(np.float32)
                            else:
                                a_tensor = local_actor(o_tensor).cpu().numpy()
                                a_tensor += np.random.normal(0, current_expl_noise, size=(N, 1))
                                joint_action = np.clip(a_tensor, 4.0, 10.0).astype(np.float32)
                        local_actor.train()

                # We subtract 4 to adapt [4, 10] to cw.cc's expected [0, 6] for "continuous"
                # since cw.cc does cw = pow(2, action + 4).
                actions_for_env = [joint_action - 4.0]

                # ----------------------------------------------------------
                # Latency: env.step() wall-clock time
                # ----------------------------------------------------------
                _t_env_start = time.time()
                next_obs_list, rewards, dones, infos = env.step(actions_for_env)
                _env_step_latency = time.time() - _t_env_start

                next_obs = next_obs_list[0]  # shape (N, 1)
                reward = rewards[0]          # scalar
                done = dones[0]

                # ----------------------------------------------------------
                # Physical metrics used for telemetry and reward shaping
                # ----------------------------------------------------------
                _info_parts = str(infos[0]).split("|")
                try:
                    sent_mbits = float(_info_parts[0])
                    throughput_mbps = sent_mbits / args.stepTime
                except (IndexError, TypeError, ValueError):
                    print(
                        f"WARNING: extra-info parse failed at step {time_step}; "
                        f"using reward fallback for throughput"
                    )
                    throughput_mbps = reward * args.nWifi * 150.0 / 10.0

                try:
                    jains_fairness_index = (
                        float(_info_parts[4]) if len(_info_parts) > 4 else float("nan")
                    )
                except (TypeError, ValueError):
                    jains_fairness_index = float("nan")

                try:
                    jains_fairness_episode = (
                        float(_info_parts[5]) if len(_info_parts) > 5 else float("nan")
                    )
                except (TypeError, ValueError):
                    jains_fairness_episode = float("nan")

                episode_rx_packets = []
                if len(_info_parts) > 6:
                    try:
                        episode_rx_packets = [
                            int(value)
                            for value in _info_parts[6].split(",")
                            if value.strip()
                        ]
                        episode_rx_packets_final = list(episode_rx_packets)
                    except (TypeError, ValueError):
                        episode_rx_packets = []

                frame_loss_rate = float(np.mean(next_obs))
                fairness_for_reward = (
                    jains_fairness_index
                    if np.isfinite(jains_fairness_index)
                    else 0.0
                )
                throughput_for_reward = np.clip(
                    throughput_mbps / max(args.ceiling_mbps, 1e-6), 0.0, 1.0
                )
                shaped_reward = (
                    0.6 * throughput_for_reward
                    + 0.3 * fairness_for_reward
                    - 0.1 * frame_loss_rate
                )
                # Use one shared cooperative reward for every agent. Repeating
                # the scalar preserves the existing joint-buffer interface.
                reward_vec = np.full(N, shaped_reward, dtype=np.float32)

                # Push joint transition to replay buffer
                replay_buffer.add(obs, joint_action, reward_vec, next_obs, done)

                obs = next_obs
                episode_reward += reward

                # 3. CTDE Training Phase
                mean_actor_loss = 0.0
                critic_loss_val = 0.0

                # ----------------------------------------------------------
                # Latency: NN update wall-clock time
                # ----------------------------------------------------------
                _t_nn_start = time.time()

                if time_step >= args.start_timesteps and len(replay_buffer) > args.batch_size:
                    jo, ja, r, jno, nd = replay_buffer.sample(args.batch_size)
                    # move to device
                    jo, ja, r, jno, nd = jo.to(device), ja.to(device), r.to(device), jno.to(device), nd.to(device)

                    # --- Critic Update ---
                    with torch.no_grad():
                        if args.use_centralized_critic:
                            # CentralizedCritic: pass full joint obs/actions
                            target_joint_action = torch.zeros_like(jno)
                            for i in range(N):
                                target_joint_action[:, i:i+1] = target_actors[i](jno[:, i:i+1])
                            target_q = target_critic(jno, target_joint_action)  # (batch, 1)
                        else:
                            # MeanFieldCritic: use mean-field approximation
                            jno_reshaped = jno.view(-1, 1)  # (batch * N, 1)
                            a_target_reshaped = target_actor(jno_reshaped)
                            target_joint_action = a_target_reshaped.view(-1, N)  # (batch, N)

                            sum_jno = jno.sum(dim=1, keepdim=True)
                            sum_ja_tgt = target_joint_action.sum(dim=1, keepdim=True)

                            mean_jno = (sum_jno - jno) / (N - 1)  # (batch, N)
                            mean_ja_tgt = (sum_ja_tgt - target_joint_action) / (N - 1)  # (batch, N)

                            jno_flat = jno.view(-1, 1)
                            a_tgt_flat = target_joint_action.view(-1, 1)
                            mean_jno_flat = mean_jno.view(-1, 1)
                            mean_ja_tgt_flat = mean_ja_tgt.view(-1, 1)

                            target_q = target_critic(jno_flat, a_tgt_flat, mean_jno_flat, mean_ja_tgt_flat)  # (batch*N, 1)

                        r_flat = r.view(-1, 1)
                        nd_flat = nd.repeat(1, N).view(-1, 1)

                        # Correct Bellman target: nd_flat is (1 - done), so multiply directly
                        if args.use_centralized_critic:
                            y = r_flat + args.discount * target_q.repeat(1, N).view(-1, 1) * nd_flat
                        else:
                            y = r_flat + args.discount * target_q * nd_flat

                    if args.use_centralized_critic:
                        # CentralizedCritic: pass full joint obs/actions
                        current_q = local_critic(jo, ja)  # (batch, 1)
                    else:
                        # MeanFieldCritic: use mean-field approximation
                        sum_jo = jo.sum(dim=1, keepdim=True)
                        sum_ja = ja.sum(dim=1, keepdim=True)
                        mean_jo = (sum_jo - jo) / (N - 1)
                        mean_ja = (sum_ja - ja) / (N - 1)

                        jo_flat = jo.view(-1, 1)
                        ja_flat = ja.view(-1, 1)
                        mean_jo_flat = mean_jo.view(-1, 1)
                        mean_ja_flat = mean_ja.view(-1, 1)

                        current_q = local_critic(jo_flat, ja_flat, mean_jo_flat, mean_ja_flat)

                    # Critic Loss
                    if args.use_centralized_critic:
                        critic_loss = F.mse_loss(current_q.repeat(1, N).view(-1, 1), y)
                    else:
                        critic_loss = F.mse_loss(current_q, y)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_critic.parameters(), max_norm=1.0)
                    critic_optimizer.step()

                    critic_loss_val = critic_loss.item()

                    # --- Actor Update ---
                    if args.use_centralized_critic:
                        # N separate actors with gradient isolation
                        # The critic is used to differentiate actor actions, but
                        # its parameters do not need gradients during this phase.
                        for parameter in local_critic.parameters():
                            parameter.requires_grad_(False)
                        for i in range(N):
                            actor_optimizers[i].zero_grad()

                            # Build predicted joint action: only actor i is "live"
                            pred_actions = []
                            for j in range(N):
                                if i == j:
                                    pred_actions.append(local_actors[j](jo[:, j:j+1]))
                                else:
                                    pred_actions.append(ja[:, j:j+1].detach())

                            new_joint_action = torch.cat(pred_actions, dim=1)  # (batch, N)
                            actor_q = local_critic(jo, new_joint_action)  # (batch, 1)

                            actor_loss = -actor_q.mean()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(local_actors[i].parameters(), max_norm=1.0)
                            actor_optimizers[i].step()
                            mean_actor_loss += actor_loss.item()
                        for parameter in local_critic.parameters():
                            parameter.requires_grad_(True)
                        mean_actor_loss /= N
                    else:
                        # Shared actor (legacy)
                        actor_optimizer.zero_grad()

                        jo_reshaped = jo.view(-1, 1)  # (batch * N, 1)
                        a_pred_reshaped = local_actor(jo_reshaped)
                        new_joint_action = a_pred_reshaped.view(-1, N)  # (batch, N)

                        sum_new_ja = new_joint_action.sum(dim=1, keepdim=True)
                        mean_new_ja = (sum_new_ja - new_joint_action) / (N - 1)

                        a_pred_flat = new_joint_action.view(-1, 1)
                        mean_new_ja_flat = mean_new_ja.view(-1, 1)

                        # Pass constructed joint action to local Mean Field Critic
                        actor_q = local_critic(jo_flat, a_pred_flat, mean_jo_flat, mean_new_ja_flat)

                        # Policy loss is negative mean of Critic's output
                        actor_loss = -actor_q.mean()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(local_actor.parameters(), max_norm=1.0)
                        actor_optimizer.step()

                        mean_actor_loss = actor_loss.item()

                    # --- Soft Updates ---
                    if args.use_centralized_critic:
                        for i in range(N):
                            soft_update(target_actors[i], local_actors[i], args.tau)
                        soft_update(target_critic, local_critic, args.tau)
                    else:
                        soft_update(target_actor, local_actor, args.tau)
                        soft_update(target_critic, local_critic, args.tau)

                _nn_update_latency = time.time() - _t_nn_start

                # Total latency = env step + NN update
                _step_latency_sec = _env_step_latency + _nn_update_latency

                time_step += 1

                # ==============================================================
                # 4. Comprehensive Telemetry & Physical Metrics
                # ==============================================================
                _raw_weights = np.maximum(1.0 - obs.flatten(), 1e-6)
                _x = (_raw_weights / _raw_weights.sum()) * throughput_mbps
                _sum_x  = float(np.sum(_x))
                _sum_x2 = float(np.sum(_x ** 2))
                if _sum_x2 > 0.0:
                    jains_fairness_derived = (_sum_x ** 2) / (N * _sum_x2)
                else:
                    jains_fairness_derived = 1.0

                episode_fairness.append(jains_fairness_index)
                episode_fairness_episode.append(jains_fairness_episode)
                episode_fairness_derived.append(jains_fairness_derived)

                # Accumulate for episode average
                episode_throughputs.append(throughput_mbps)
                episode_actions.append(float(np.mean(joint_action)))

                # Per-step diagnostic JSONL (obs per station, joint action, throughput)
                if args.jsonl:
                    with open(args.jsonl, "a") as f:
                        f.write(json.dumps({
                            "episode": episode,
                            "step": time_step,
                            "obs": obs.flatten().tolist(),
                            "joint_action": joint_action.flatten().tolist(),
                            "throughput_mbps": throughput_mbps,
                            "reward": float(reward),
                            "done": bool(done),
                            "jains_fairness_index": jains_fairness_index,
                            "jains_fairness_episode": jains_fairness_episode,
                            "jains_fairness_derived": jains_fairness_derived,
                        }) + "\n")

                # 4e. Convergence tracker (rolling window = 100 steps)
                _throughput_window.append(throughput_mbps)
                _rolling_avg = float(np.mean(_throughput_window))

                if time_step > args.start_timesteps:

                    _convergence_payload: dict = {}
                    if (
                        not _convergence_logged
                        and _rolling_avg > args.convergence_frac * args.ceiling_mbps
                    ):
                        _convergence_payload["convergence_epoch"] = time_step
                        _convergence_epoch = time_step
                        _convergence_logged = True

                    # ----------------------------------------------------------
                    # 4f. Fire the wandb.log call with all metrics
                    # ----------------------------------------------------------
                    _log_payload: dict = {
                        "step":                      time_step,
                        # ---- core losses / rewards ----
                        "global_reward":             reward,
                        "critic_loss":               critic_loss_val,
                        "mean_actor_loss":           mean_actor_loss,
                        "episode":                   episode,
                        # ---- latency ----
                        "step_execution_latency_sec": _step_latency_sec,
                        # ---- network state ----
                        "frame_loss_rate":           frame_loss_rate,
                        "instantaneous_throughput_mbps": throughput_mbps,
                        # ---- fairness ----
                        "jains_fairness_index":      jains_fairness_index,
                        "jains_fairness_episode":    jains_fairness_episode,
                        "jains_fairness_derived":     jains_fairness_derived,
                        # ---- CW exponent distribution ----
                        "station_cw_distribution":   wandb.Histogram(
                            joint_action.flatten().tolist()
                        ),
                        # ---- rolling convergence avg ----
                        "throughput_rolling_avg_mbps": _rolling_avg,
                        **_convergence_payload,
                    }
                    wandb.log(_log_payload)

                    t.set_postfix(
                        reward=f"{reward:.2f}",
                        q_loss=f"{critic_loss_val:.3f}",
                        pi_loss=f"{mean_actor_loss:.3f}",
                        tput=f"{throughput_mbps:.2f}Mb/s",
                        fairness=f"{jains_fairness_index:.3f}",
                    )

                if done:
                    break

        # ==================================================================
        # 5. Episode-boundary telemetry
        # ==================================================================
        current_expl_noise = max(args.expl_noise_min, current_expl_noise * args.expl_noise_decay)
        
        mean_ep_throughput = float(np.mean(episode_throughputs)) if episode_throughputs else 0.0
        mean_ep_action = float(np.mean(episode_actions)) if episode_actions else 0.0
        mature_episode_fairness = (
            episode_fairness[100:] if len(episode_fairness) > 100
            else []
        )
        mean_ep_fairness = (
            float(np.nanmean(mature_episode_fairness))
            if mature_episode_fairness else float("nan")
        )
        final_ep_fairness = (
            episode_fairness_episode[-1] if episode_fairness_episode else float("nan")
        )
        mean_ep_fairness_derived = (
            float(np.nanmean(episode_fairness_derived))
            if episode_fairness_derived else float("nan")
        )
        _all_episode_throughputs.append(mean_ep_throughput)
        _all_episode_actions.append(mean_ep_action)
        _all_episode_fairness.append(final_ep_fairness)
        _all_episode_fairness_rolling.append(mean_ep_fairness)
        _all_episode_fairness_derived.append(mean_ep_fairness_derived)
        _all_episode_rx_packets.append(list(episode_rx_packets_final))
        wandb.log({
            "step":                      time_step,
            "mean_episode_throughput_mbps": mean_ep_throughput,
            "episode_total_reward":      episode_reward,
            "jains_fairness_episode":    final_ep_fairness,
            "mean_jains_fairness_index": mean_ep_fairness,
            "mean_jains_fairness_derived": mean_ep_fairness_derived,
        })
        if args.jsonl:
            with open(args.jsonl, "a") as f:
                f.write(json.dumps({
                    "episode": episode,
                    "episode_end": True,
                    "step": time_step,
                    "jains_fairness_episode": final_ep_fairness,
                    "mean_jains_fairness_index": mean_ep_fairness,
                    "mean_jains_fairness_derived": mean_ep_fairness_derived,
                    "episode_rx_packets": list(episode_rx_packets_final),
                }) + "\n")

        print(
            f"Episode {episode} | Reward: {episode_reward:.3f} "
            f"| Mean Throughput: {mean_ep_throughput:.2f} Mb/s"
        )
        env.close()

    # Final per-episode summary + config for benchmarking
    if args.summary_json:
        summary = {
            "config": {
                "nWifi": N, "simTime": args.simTime, "stepTime": args.stepTime,
                "episodes": args.episodes, "seed": args.seed,
                "start_timesteps": args.start_timesteps, "batch_size": args.batch_size,
                "discount": args.discount, "tau": args.tau, "buffer_size": args.buffer_size,
                "expl_noise": args.expl_noise, "logit_noise": args.logit_noise,
                "use_centralized_critic": args.use_centralized_critic,
                "ceiling_mbps": args.ceiling_mbps,
                "convergence_frac": args.convergence_frac,
                "convergence_threshold_mbps":
                    args.convergence_frac * args.ceiling_mbps,
                "convergence_epoch": _convergence_epoch,
            },
            "episodes": [
                {
                    "episode": e,
                    "mean_throughput_mbps": tp,
                    "mean_action": ac,
                    "jains_fairness_episode": ep_fairness,
                    "mean_jains_fairness_index": rolling_fairness,
                    "mean_jains_fairness_derived": derived_fairness,
                    "episode_rx_packets": rx_packets,
                }
                for e, tp, ac, ep_fairness, rolling_fairness, derived_fairness, rx_packets in zip(
                    range(len(_all_episode_throughputs)),
                    _all_episode_throughputs,
                    _all_episode_actions,
                    _all_episode_fairness,
                    _all_episode_fairness_rolling,
                    _all_episode_fairness_derived,
                    _all_episode_rx_packets,
                )
            ],
            "final_mean_throughput_mbps": mean_ep_throughput,
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    run.finish()
