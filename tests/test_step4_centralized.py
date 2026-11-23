#!/usr/bin/env python3
"""
STEP 4 — Verify the N-separate-actors + CentralizedCritic architecture.

Checks:
  T1. CentralizedCritic consumes joint obs + joint actions (dim 2N -> 1).
  T2. Actor gradient isolation: during actor i's update, only actor i
      receives gradient; all other actors' .grad stay None.
  T3. Gradients differ across stations (heterogeneous local obs).
  T4. CentralizedCritic sees cross-agent information: changing agent j's
      action alone changes Q(i) -> the critic is not mean-field limited.

Run:  python3 tests/test_step4_centralized.py       (CPU only, no ns-3)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from agents.maddpg.model import (
    Actor, CentralizedCritic, create_homogeneous_actors,
)

SEP = "=" * 70
results = {}


def cpu_actors(n, seed=42):
    """create_homogeneous_actors puts actors on the global device; force CPU."""
    return [a.to("cpu") for a in create_homogeneous_actors(n, 1, 1, seed=seed)]


# -------------------------------------------------------------------------
# T1 — critic input dimensionality
# -------------------------------------------------------------------------
def test_t1_critic_dims():
    print(f"\n{SEP}\nT1 — CentralizedCritic input = joint obs + joint actions (2N)\n{SEP}")
    N = 5
    critic = CentralizedCritic(N, 1, 1, seed=0).to("cpu")
    jo = torch.rand(8, N)
    ja = torch.rand(8, N)
    q = critic(jo, ja)
    ok_shape = tuple(q.shape) == (8, 1)
    in_features = critic.fc1.in_features
    ok_dim = in_features == 2 * N
    print(f"  fc1.in_features = {in_features}   (expected {2*N})")
    print(f"  Q output shape  = {tuple(q.shape)}   (expected (8, 1))")
    print(f"  Result: {'PASS' if ok_shape and ok_dim else 'FAIL'}")
    return ok_shape and ok_dim


# -------------------------------------------------------------------------
# T2 — gradient isolation
# -------------------------------------------------------------------------
def test_t2_gradient_isolation():
    print(f"\n{SEP}\nT2 — Actor gradient isolation (only actor i gets gradient)\n{SEP}")
    torch.manual_seed(0)
    N, batch = 5, 32
    actors = cpu_actors(N)
    critic = CentralizedCritic(N, 1, 1, seed=0).to("cpu")
    opts = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in actors]

    jo = torch.rand(batch, N)
    ja = torch.rand(batch, N)

    all_ok = True
    for i in range(N):
        for a in actors:
            a.zero_grad()
        opts[i].zero_grad()

        pred = []
        for j in range(N):
            if j == i:
                pred.append(actors[j](jo[:, j:j+1]))          # live
            else:
                pred.append(ja[:, j:j+1].detach())            # stop-grad
        new_ja = torch.cat(pred, dim=1)

        loss = -critic(jo, new_ja).mean()
        loss.backward()

        got = [any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in actors[j].parameters()) for j in range(N)]
        others_clean = all(not got[j] for j in range(N) if j != i)
        self_has_grad = got[i]
        all_ok &= others_clean and self_has_grad
        print(f"  actor {i} update -> grad present on: "
              f"{[j for j in range(N) if got[j]]}  "
              f"({'OK' if others_clean and self_has_grad else 'LEAK'})")

    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# -------------------------------------------------------------------------
# T3 — gradients differ across stations
# -------------------------------------------------------------------------
def test_t3_distinct_gradients():
    print(f"\n{SEP}\nT3 — Per-station gradients differ under heterogeneous obs\n{SEP}")
    torch.manual_seed(0)
    N, batch = 5, 64
    actors = cpu_actors(N)
    critic = CentralizedCritic(N, 1, 1, seed=0).to("cpu")

    # Heterogeneous local observations: station i sees a distinct loss regime
    jo = torch.linspace(0.05, 0.95, N).unsqueeze(0).repeat(batch, 1)
    ja = torch.rand(batch, N) * 6.0 + 4.0

    grads = []
    for i in range(N):
        for a in actors:
            a.zero_grad()
        pred = [actors[j](jo[:, j:j+1]) if j == i else ja[:, j:j+1].detach()
                for j in range(N)]
        new_ja = torch.cat(pred, dim=1)
        (-critic(jo, new_ja).mean()).backward()
        grads.append(torch.cat([p.grad.flatten() for p in actors[i].parameters()
                                if p.grad is not None]).clone())

    max_diff = max((grads[i] - grads[j]).norm().item()
                   for i in range(N) for j in range(i + 1, N))
    norms = [g.norm().item() for g in grads]
    print("  per-actor grad norms: " + ", ".join(f"{n:.6e}" for n in norms))
    print(f"  max pairwise grad difference: {max_diff:.6e}")
    ok = max_diff > 1e-8
    print(f"  Result: {'PASS — gradients differ across stations' if ok else 'FAIL — identical'}")
    return ok


# -------------------------------------------------------------------------
# T4 — critic sees cross-agent actions (refutes mean-field equivalence)
# -------------------------------------------------------------------------
def test_t4_cross_agent_coupling():
    print(f"\n{SEP}\nT4 — Q responds to a single other agent's action\n{SEP}")
    torch.manual_seed(0)
    N = 5
    critic = CentralizedCritic(N, 1, 1, seed=0).to("cpu")
    jo = torch.full((1, N), 0.3)
    ja = torch.full((1, N), 5.0)

    q_base = critic(jo, ja).item()

    max_sens = 0.0
    for j in range(N):
        ja2 = ja.clone()
        ja2[0, j] = 9.0         # perturb ONE agent's action
        d = abs(critic(jo, ja2).item() - q_base)
        max_sens = max(max_sens, d)
        print(f"  perturb agent {j}: |dQ| = {d:.6e}")

    # Mean-field Q_i only sees the *mean* of others, so a permutation of
    # other agents' actions is invisible to it. True centralized Q should
    # distinguish them. Demonstrate the critic is not permutation-collapsed:
    ja_a = torch.tensor([[5.0, 4.0, 6.0, 4.0, 6.0]])
    ja_b = torch.tensor([[5.0, 6.0, 4.0, 6.0, 4.0]])  # same mean(others)=5.0
    d_perm = abs(critic(jo, ja_a).item() - critic(jo, ja_b).item())
    print(f"  same-mean permutation |dQ| = {d_perm:.6e}  (mean-field would be 0)")
    ok = max_sens > 1e-6 and d_perm > 1e-8
    print(f"  Result: {'PASS — critic consumes full joint action' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results["T1_critic_dims"] = test_t1_critic_dims()
    results["T2_grad_isolation"] = test_t2_gradient_isolation()
    results["T3_distinct_grads"] = test_t3_distinct_gradients()
    results["T4_cross_agent"] = test_t4_cross_agent_coupling()

    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    sys.exit(0 if all(results.values()) else 1)
