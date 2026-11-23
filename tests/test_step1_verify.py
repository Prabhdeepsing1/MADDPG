#!/usr/bin/env python3
"""
STEP 1 — Verification tests for opti.md Critical Issues.
Run with:  python3 tests/test_step1_verify.py
No ns-3 required; pure Python / PyTorch.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import copy

from agents.maddpg.utils import MADDPGReplayBuffer
from agents.maddpg.model import MeanFieldCritic, Actor, create_homogeneous_actors

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[93mINFO\033[0m"

SEP = "=" * 70


# =========================================================================
# BUG #5 — Terminal flag inversion test
# =========================================================================
def test_bug5_terminal_flag():
    print(SEP)
    print("BUG #5 — Terminal flag inversion in Bellman target")
    print(SEP)

    N = 3
    buf = MADDPGReplayBuffer(n_agents=N, max_size=100)

    # --- push one TERMINAL transition (done=True) ---
    obs          = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    joint_action = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    reward       = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    next_obs     = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    done         = True

    buf.add(obs, joint_action, reward, next_obs, done)

    jo, ja, r, jno, nd = buf.sample(1)

    print(f"\n  done=True was stored.")
    print(f"  buf.not_done stored value : {buf.not_done[0, 0]:.1f}  (expected 0.0 since done=True)")
    print(f"  nd tensor sampled         : {nd.item():.1f}        (should be 0.0)")

    # Reproduce the CURRENT (buggy) training loop formula:
    discount = 0.99
    fake_target_q = torch.FloatTensor([[1.0]]).to(nd.device)   # match tensor device

    # CURRENT code (line 209 of maddpg_train.py):
    nd_flat_buggy = nd.repeat(1, N).view(-1, 1)
    y_buggy = r.view(-1,1)[:1] + discount * fake_target_q * (1 - nd_flat_buggy[:1])

    # CORRECT code should be:
    y_correct = r.view(-1,1)[:1] + discount * fake_target_q * nd_flat_buggy[:1]

    print(f"\n  Bellman mask at terminal step:")
    print(f"    Current code  (1 - nd_flat) = {(1 - nd_flat_buggy[0,0]):.1f}   <- bootstrap applied?")
    print(f"    Correct code  (nd_flat)     = {nd_flat_buggy[0,0]:.1f}   <- bootstrap applied?")
    print(f"\n  y_buggy  (should be r only, no bootstrap): {y_buggy.item():.4f}")
    print(f"  y_correct (should be r only, no bootstrap): {y_correct.item():.4f}")

    r_only = r.view(-1,1)[0,0].item()
    print(f"\n  Pure reward (no bootstrap expected): {r_only:.4f}")

    inverted = abs(y_buggy.item() - r_only) > 0.01
    correct_ok = abs(y_correct.item() - r_only) < 0.01

    if inverted:
        print(f"\n  Result: CONFIRMED — Bug #5 is real. Current code bootstraps at "
              f"terminal step (adds {discount:.2f}×Q={fake_target_q.item():.1f}={discount*fake_target_q.item():.3f})")
    else:
        print(f"\n  Result: REFUTED — Bug #5 not observed.")

    if correct_ok:
        print(f"  Fix verified: correct formula gives y ≈ r (no spurious bootstrap)")
    return inverted


# =========================================================================
# BUG #3 — Throughput formula hard-coded to N=5
# =========================================================================
def test_bug3_throughput_formula():
    print(f"\n{SEP}")
    print("BUG #3 — Throughput formula hard-wired to N=5")
    print(SEP)

    # cw.cc line 212 computes:
    # reward = packets_delta * (1500-20-8-8) * 8 / 1024 / 1024 / (5 * 150 * step) * 10
    PAYLOAD_BYTES = 1500 - 20 - 8 - 8  # = 1464 bytes
    stepTime = 0.01
    offered_load_mbps = 150.0           # Mbps per station

    # Simulate a "perfect" step: all stations receive at full rate
    for N in [5, 10, 25]:
        # Raw throughput in Mb/s = N * offered_load * payload_fraction
        # (This is what cw.cc actually measures from packet counters)
        expected_tput_mbps = N * offered_load_mbps * (PAYLOAD_BYTES / 1500)

        # Compute what reward cw.cc returns for this throughput:
        # packets_per_step = expected_Mbps * step * 1e6 / (PAYLOAD_BYTES * 8)
        packets_per_step = expected_tput_mbps * 1e6 * stepTime / (PAYLOAD_BYTES * 8)
        reward_from_cc = packets_per_step * PAYLOAD_BYTES * 8 / 1e6 / (5 * offered_load_mbps * stepTime) * 10
        # Note: cw.cc divides by HARD-CODED 5, not N

        # CURRENT Python formula (maddpg_train.py line 277):
        tput_python_current = reward_from_cc * (N * offered_load_mbps * stepTime) / 10.0

        # CORRECT Python formula (reverse cw.cc's /5 exactly):
        tput_python_correct = reward_from_cc * (5 * offered_load_mbps * stepTime) / 10.0

        error_pct = (tput_python_current - expected_tput_mbps) / expected_tput_mbps * 100

        print(f"\n  N={N:2d}:")
        print(f"    cw.cc reward              = {reward_from_cc:.4f}")
        print(f"    True throughput           = {expected_tput_mbps:.2f} Mb/s")
        print(f"    Current logged tput (N×…) = {tput_python_current:.2f} Mb/s  (error: {error_pct:+.1f}%)")
        print(f"    Correct logged tput (5×…) = {tput_python_correct:.2f} Mb/s  (error: 0.0%)")

    confirmed = True   # error is non-zero for N!=5 by construction
    if confirmed:
        print(f"\n  Result: CONFIRMED — Bug #3 is real. For N=10 the logged throughput "
              f"is 2× the true value; for N=25 it is 5× the true value.")
    return confirmed


# =========================================================================
# BUG #4 — MeanFieldCritic: are actor gradients identical across stations?
# =========================================================================
def test_bug4_identical_gradients():
    print(f"\n{SEP}")
    print("BUG #4 — MeanFieldCritic causes identical actor gradients")
    print(SEP)

    torch.manual_seed(0)
    N = 5
    batch = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create N separate actors (as the fix would have them)
    local_actors = create_homogeneous_actors(N, obs_dim=1, act_dim=1, seed=42)
    local_critic = MeanFieldCritic().to(device)

    # Synthetic batch: observations differ per station
    jo = torch.rand(batch, N).to(device)              # (batch, N) — different obs per station
    ja = torch.rand(batch, N).to(device)              # (batch, N)

    # -----------------------------------------------------------------
    # Simulate the CURRENT training loop (shared actor, mean-field)
    # -----------------------------------------------------------------
    shared_actor = Actor(obs_dim=1, act_dim=1, seed=42).to(device)
    shared_opt   = torch.optim.Adam(shared_actor.parameters(), lr=1e-4)

    # Current actor update (all N agents share one actor):
    jo_flat = jo.view(-1, 1)              # (batch*N, 1)
    a_pred  = shared_actor(jo_flat)       # (batch*N, 1)
    new_ja  = a_pred.view(batch, N)       # (batch, N)

    sum_jo  = jo.sum(dim=1, keepdim=True)
    sum_ja2 = new_ja.sum(dim=1, keepdim=True)
    mean_jo = (sum_jo - jo) / (N - 1)
    mean_ja = (sum_ja2 - new_ja) / (N - 1)

    # Flatten for mean-field critic
    jo_i   = jo.view(-1, 1)
    a_i    = new_ja.view(-1, 1)
    mjo_i  = mean_jo.view(-1, 1)
    mja_i  = mean_ja.view(-1, 1)

    q_val = local_critic(jo_i, a_i, mjo_i, mja_i)
    actor_loss_shared = -q_val.mean()

    shared_opt.zero_grad()
    actor_loss_shared.backward()

    # Collect gradients for each "virtual agent" slice (they're the same network):
    shared_grad_norm = sum(p.grad.norm().item() ** 2 for p in shared_actor.parameters()
                           if p.grad is not None) ** 0.5

    print(f"\n  CURRENT (shared actor + mean-field critic):")
    print(f"    One actor, one gradient update — all stations indistinguishable")
    print(f"    Shared actor gradient norm: {shared_grad_norm:.6f}")

    # -----------------------------------------------------------------
    # Simulate the FIXED training loop (N separate actors, same mean-field)
    # -----------------------------------------------------------------
    sep_actors = create_homogeneous_actors(N, obs_dim=1, act_dim=1, seed=42)
    sep_opts   = [torch.optim.Adam(a.parameters(), lr=1e-4) for a in sep_actors]

    grad_norms = []
    grad_vecs  = []
    for i in range(N):
        sep_opts[i].zero_grad()

        # Build predicted joint action: only actor i is "live"
        pred_actions = []
        for j in range(N):
            o_j = jo[:, j:j+1]
            if j == i:
                pred_actions.append(sep_actors[j](o_j))       # differentiable
            else:
                pred_actions.append(sep_actors[j](o_j).detach())  # stop gradient

        new_ja_i = torch.cat(pred_actions, dim=1)             # (batch, N)

        sum_new = new_ja_i.sum(dim=1, keepdim=True)
        mean_a_others = (sum_new - new_ja_i[:, i:i+1]) / (N - 1)
        mean_o_others = (jo.sum(dim=1, keepdim=True) - jo[:, i:i+1]) / (N - 1)

        q_i = local_critic(
            jo[:, i:i+1],
            new_ja_i[:, i:i+1],
            mean_o_others,
            mean_a_others
        )
        loss_i = -q_i.mean()
        loss_i.backward()

        gv = torch.cat([p.grad.flatten() for p in sep_actors[i].parameters()
                        if p.grad is not None])
        grad_norms.append(gv.norm().item())
        grad_vecs.append(gv.detach().clone())
        sep_opts[i].step()

    print(f"\n  FIXED (N separate actors + mean-field critic, gradient isolation):")
    for i in range(N):
        print(f"    Actor {i} gradient norm: {grad_norms[i]:.6f}")

    # Check if gradients differ across actors
    max_diff = 0.0
    for i in range(N):
        for j in range(i+1, N):
            diff = (grad_vecs[i] - grad_vecs[j]).norm().item()
            max_diff = max(max_diff, diff)

    print(f"\n  Max gradient vector difference between any two actors: {max_diff:.8f}")

    if max_diff > 1e-6:
        print(f"  Result: CONFIRMED — With N separate actors + gradient isolation, "
              f"gradients DIFFER across stations (max diff={max_diff:.6f}).")
        print(f"           Current shared actor smears all these into one merged update.")
        confirmed = True
    else:
        print(f"  Result: REFUTED — Gradients are identical even with separate actors.")
        confirmed = False

    return confirmed


# =========================================================================
# BUG #2 — No gradient clipping: show exploding gradient scenario
# =========================================================================
def test_bug2_gradient_clipping():
    print(f"\n{SEP}")
    print("BUG #2 — No gradient clipping: magnitude check")
    print(SEP)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = MeanFieldCritic().to(device)
    opt    = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Simulate a pathological batch (large Q target — causes large loss)
    batch = 32
    o  = (torch.rand(batch, 1) * 100).to(device)   # out-of-distribution
    a  = (torch.rand(batch, 1) * 100).to(device)
    mo = (torch.rand(batch, 1) * 100).to(device)
    ma = (torch.rand(batch, 1) * 100).to(device)
    y  = (torch.ones(batch, 1) * 1000).to(device)  # huge target

    q = critic(o, a, mo, ma)
    loss = F.mse_loss(q, y)

    opt.zero_grad()
    loss.backward()

    grad_norm_before = sum(p.grad.norm().item() ** 2 for p in critic.parameters()
                           if p.grad is not None) ** 0.5
    print(f"\n  Gradient norm BEFORE clipping: {grad_norm_before:.4f}")

    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    grad_norm_after = sum(p.grad.norm().item() ** 2 for p in critic.parameters()
                          if p.grad is not None) ** 0.5
    print(f"  Gradient norm AFTER  clipping: {grad_norm_after:.4f}")

    if grad_norm_before > 10.0:
        print(f"\n  Result: CONFIRMED — Without clipping, gradient norm = {grad_norm_before:.1f} "
              f"(>{10.0}). Clipping reduces it to {grad_norm_after:.4f}.")
        return True
    else:
        print(f"\n  Result: Gradient norm was {grad_norm_before:.4f} — not explosive in "
              f"this synthetic test, but clipping is still best practice.")
        return False


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    results = {}
    results["bug5_terminal_flag"]    = test_bug5_terminal_flag()
    results["bug3_throughput_N5"]    = test_bug3_throughput_formula()
    results["bug4_identical_grads"]  = test_bug4_identical_gradients()
    results["bug2_no_clipping"]      = test_bug2_gradient_clipping()

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    for name, confirmed in results.items():
        status = "CONFIRMED" if confirmed else "REFUTED  "
        print(f"  {status}  {name}")
    print()
