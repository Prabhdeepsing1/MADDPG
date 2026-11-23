"""
diag/cw_mapping_check.py
Verify the action-to-CW mapping chain end to end.
"""
import numpy as np

print("=== Action-to-CW Mapping Chain ===\n")

# Step 1: Actor output range
print("Step 1: Actor forward() output")
print("  Actor uses: sigmoid(fc3) scaled to [ACTION_LOW, ACTION_HIGH] = [4.0, 10.0]")
print("  Output range: a_i ∈ [4.0, 10.0]  (strictly, never exactly 4 or 10 due to sigmoid)")
print()

# Step 2: maddpg_train.py actions_for_env
print("Step 2: maddpg_train.py line 142")
print("  actions_for_env = joint_action - 4.0")
print("  i.e., subtracts 4 from each action before sending to env")
print("  So env receives: a'_i = a_i - 4.0  ∈ [0.0, 6.0]")
print()

# Step 3: cw.cc MyExecuteActions, type="continuous" (line 170-172)
print("Step 3: cw.cc MyExecuteActions (type='continuous'), lines 170-172")
print("  cw_val = pow(2, actionVector[i] + 4)")
print("  env receives a'_i ∈ [0, 6] → cw_val = 2^(a'_i + 4) = 2^a_i (since a'_i = a_i - 4)")
print()

# Verify round-trip
for a_i in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    a_prime = a_i - 4.0
    cw_val = 2 ** (a_prime + 4)
    print(f"  actor output a_i={a_i:.1f} → env gets {a_prime:.1f} → CW = 2^{a_i:.1f} = {cw_val:.0f}")

print()
print("Step 4: cw.cc line 183: clamp to [min_cw=16, max_cw=1024]")
print("  At a_i=4.0: CW = 2^4 = 16   (at min_cw=16)")
print("  At a_i=10.0: CW = 2^10 = 1024 (at max_cw=1024)")
print("  So effective range is CW ∈ [16, 1024]")
print()

# Step 5: BEB check
print("=== BEB (Binary Exponential Backoff) Status ===")
print("cw.cc lines 186-189: sets BOTH MinCw AND MaxCw to cw_val")
print("  Config::Set(pathMin, UintegerValue(cw_val))")
print("  Config::Set(pathMax, UintegerValue(cw_val))")
print()
print("→ MinCw == MaxCw: BEB is DISABLED. After each collision the MAC layer")
print("  doubles CW up to MaxCw. Since MaxCw == MinCw, doubling has no effect;")
print("  CW stays fixed at cw_val. BEB is effectively suppressed per-station. ✓")
print()

# Step 6: Random exploration action
print("=== Warm-up / Random Exploration (time_step < start_timesteps=300) ===")
print("maddpg_train.py line 127:")
print("  joint_action = np.random.uniform(4.0, 10.0, size=(N, 1))")
print("  actions_for_env = joint_action - 4.0  → env receives U[0, 6]")
print("  → CW = 2^(U[0,6]+4) = 2^U[4,10]")
print()
for _ in range(5):
    a = np.random.uniform(4.0, 10.0)
    cw = int(2**a)
    cw = min(1024, max(16, cw))
    print(f"  random a={a:.3f} → CW={cw}")

print()
print("=== maddpg_train obs → reward_vec Credit Assignment ===")
print("Lines 158-164:")
print("  alpha = 0.5")
print("  local_success = 1 - obs  (obs is frame-loss rate ρ_i, so 1-ρ is success rate)")
print("  norm_local_success = local_success / (mean + 1e-6)")
print("  reward_vec[i] = 0.5 * global_reward + 0.5 * global_reward * norm_local_success[i]")
print("  = global_reward * (0.5 + 0.5 * norm_local_success[i])")
print()
print("NOTE: The stored reward_vec is per-station (N,) shaped.")
print("      But the buffer stores it as (N,) and the critic uses it flattened.")
print("      This means the critic's target y uses per-agent shaped rewards,")
print("      which is correct for MADDPG but mixes global and local signals.")

print()
print("=== Observation Feature Description ===")
print("cw.cc MyGetObservation() lines 244-259:")
print("  For each station i:")
print("    received = g_rxPktNum[i] - last_rx[i]  (packets received at AP from STA i this step)")
print("    sent     = g_txPktNum[i] - last_tx[i]  (packets sent by STA i at PHY layer)")
print("    errs     = sent - received")
print("    ratio    = errs / sent  (collision/loss rate, range [0, 1])")
print("  Observation: ρ_i = frame-loss rate for station i")
print()
print("  NOTE: g_txPktNum counts PhyTxBegin (all transmissions including retransmissions).")
print("        g_rxPktNum counts Rx at UdpServer (payload level).")  
print("        So ratio includes retransmissions as 'errors', giving PACKET LOSS + COLLISION RATE.")

