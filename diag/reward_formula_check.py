"""
diag/reward_formula_check.py
Verify the reward formula in cw.cc and the throughput back-calculation in maddpg_train.py.
No environment is started; all computation is pure Python math.
"""
import numpy as np

N_paper = 40  # nWifi used in runs
envStepTime = 0.01  # seconds
packet_payload = 1500 - 20 - 8 - 8  # bytes = 1464

# ---- cw.cc:212 reward formula ----
# reward = res * (1500-20-8-8)*8.0 / 1024/1024 / (5 * 150 * envStepTime) * 10
# where res = packets_received_this_step
# "5" is HARDCODED in the denominator — NOT nWifi.
# "150" is offeredLoad per station (Mbps) from cw.cc:467

denom_hardcoded_N = 5
offeredLoad_Mbps = 150   # per station, from cw.cc line 467

print("=== cw.cc Reward Formula Analysis ===")
print(f"Normalisation denominator N in reward: {denom_hardcoded_N}  (HARDCODED, NOT nWifi={N_paper})")
print()

# If reward == 1.0, how many Mb/s does that represent?
# reward = MB_sent / (5 * 150 * 0.01) * 10
# reward = MB_sent / 7.5 * 10 = MB_sent / 0.75
# 1.0 = MB_sent / 0.75 → MB_sent = 0.75 MB per step
# per-step = 0.01 s → throughput = 0.75 MB / 0.01 s = 75 MB/s = 600 Mb/s (WRONG aggregate)

# Correct derivation: reward == 1.0 means:
# MB = reward * 5 * 150 * 0.01 / 10  = reward * 0.75
# Mb/s = MB * 8 / envStepTime = reward * 0.75 * 8 / 0.01 = reward * 600 Mb/s  ← for N=5 normalisation
mb_per_reward_unit_denom5 = denom_hardcoded_N * offeredLoad_Mbps * envStepTime / 10  # should be 0.06 for N=5
# Wait — let me re-read: denom is (5 * 150 * 0.01) = 7.5, multiplied by 10 → normaliser = 0.75
# reward = MB_sent_step / 0.75

# So reward = 1 → MB_sent_step = 0.75 MB → aggregate throughput = 0.75*8/0.01 = 600 Mb/s (for N=5)
# But with N=40 actual stations, offered load is 40*150 = 6000 Mbps
# Max realistic throughput ≈ 600 Mb/s (for 20 MHz 802.11ax MCS11)

print("=== maddpg_train.py throughput_mbps back-calculation (line 277) ===")
print("Formula: throughput_mbps = reward * (N * 150.0 * stepTime) / 10.0")
print(f"         = reward * ({N_paper} * 150.0 * {envStepTime}) / 10.0")
print(f"         = reward * {N_paper * 150.0 * envStepTime / 10.0:.4f}")
print()

# Reward from cw.cc uses denom (5 * 150 * 0.01)/10 = 0.075
# maddpg_train uses   (N * 150 * 0.01)/10 = (40 * 150 * 0.01)/10 = 6.0
# So the scale mismatch is: 6.0 / 0.075 = 80x

cw_cc_normaliser  = denom_hardcoded_N * offeredLoad_Mbps * envStepTime / 10.0   # 0.075
oscar_normaliser  = N_paper           * offeredLoad_Mbps * envStepTime / 10.0   # 6.0

print(f"cw.cc reward normaliser (denom):   {cw_cc_normaliser}")
print(f"maddpg_train throughput scale:      {oscar_normaliser}")
print(f"Scale mismatch factor:             {oscar_normaliser / cw_cc_normaliser:.1f}x")
print()

# If the true throughput seen by sensors is T Mb/s,
# the cw.cc reward r = T / (5 * 150) = T / 750
# The maddpg_train throughput_mbps = r * 6.0 = T * 6.0 / 750 = T / 125 ...
# Wait let me be more careful.
# From cw.cc line 212:
#   reward = (pkts * 1464 * 8 / 2^20 Mb) / (5 * 150 * 0.01) * 10
#   reward = Mb_sent_in_step / (5 * 150 * 0.01 / 10)
#   reward = Mb_sent_in_step / 0.075          (for N=40 real stations, MB is aggregate of 40)
#
# maddpg_train line 277:
#   throughput_mbps = reward * (40 * 150 * 0.01) / 10 = reward * 6.0
#
# Combined: throughput_mbps = Mb_sent_step / 0.075 * 6.0 = Mb_sent_step * 80
# But Mb_sent_step / step_time = aggregate Mb/s → throughput_mbps = (Mb_sent_step/0.01) * 80 * 0.01
#                                                                   = aggregate_Mbps * 0.8 ... 
# Hmm. Let me recompute fresh:
# Let A = aggregate throughput in Mb/s (what we want to see, e.g. 38 Mb/s from paper)
# MB_sent_in_step = A * envStepTime / 8    ... No, A is already in Mb/s
# MB_in_step = A [Mb/s] * 0.01 [s] / 8 [bits/byte] ... still wrong.
# A Mb/s = A * 0.01 Mb per step. (no /8 needed since already in bits)
# So Mb_in_step = A * envStepTime = A * 0.01 Mb

# cw.cc reward:
#   reward = (A * 0.01) / 0.075 = A * 0.01 / 0.075 = A / 7.5
#
# maddpg_train throughput_mbps:
#   throughput_mbps = reward * 6.0 = (A / 7.5) * 6.0 = A * 0.8
#
# So if actual aggregate throughput A = 38 Mb/s:
#   reward ≈ 38 / 7.5 ≈ 5.07  → EXCEEDS the [0,1] clamp!
#
# Indeed cw.cc line 222: if reward > 1.0 → use last_reward (stale value)
# That means throughput above 7.5 Mb/s (for N=5 normalisation) triggers clamping.

print("=== Reward Saturation / Clamping Analysis ===")
A_paper = 38.0   # target aggregate Mb/s
reward_if_A = A_paper / (denom_hardcoded_N * offeredLoad_Mbps)  # = 38/(5*150) = 0.0507
print(f"If aggregate throughput = {A_paper} Mb/s:")
print(f"  MB_in_step = {A_paper*envStepTime:.4f} Mb")
print(f"  cw.cc raw reward = {A_paper*envStepTime / cw_cc_normaliser:.4f}")
print(f"  Reward > 1.0? {A_paper*envStepTime / cw_cc_normaliser > 1.0}")
print()

# clamping threshold: reward > 1 when MB_in_step > 0.075
# → aggregate > 0.075 / 0.01 = 7.5 Mb/s
saturation_threshold_mbps = cw_cc_normaliser / envStepTime
print(f"Reward saturates (> 1.0) when aggregate throughput > {saturation_threshold_mbps:.1f} Mb/s")
print(f"This means for N=40 stations, reward is ALWAYS clamped to last_reward above {saturation_threshold_mbps:.1f} Mb/s!")
print()

# ---- What does reward≈0.5 (observed) mean? ----
observed_reward = 0.48
actual_mbps = observed_reward * cw_cc_normaliser / envStepTime
oscar_reported = observed_reward * oscar_normaliser
print(f"=== Observed reward ≈ {observed_reward} ===")
print(f"  Actual aggregate throughput implied: {actual_mbps:.2f} Mb/s")
print(f"  maddpg_train reports throughput:      {oscar_reported:.2f} Mb/s")
print(f"  Discrepancy: maddpg_train overestimates by factor {oscar_normaliser/cw_cc_normaliser:.1f}x")
print()

# ---- The N=5 vs N=40 mismatch in the reward denominator ----
print("=== Summary of Bugs Found ===")
print("BUG 1 (CRITICAL): cw.cc:212 reward denominator hardcodes N=5")
print("  ('5 * 150 * envStepTime') regardless of --nWifi argument.")
print("  At nWifi=40, this means the reward is scaled as if there were only 5 stations.")
print("  Practical effect: reward never accurately reflects 40-station throughput.")
print()
print("BUG 2 (CRITICAL): cw.cc:222 clamps reward to [0, 1] using stale last_reward.")
print("  For N=40, any aggregate throughput > 7.5 Mb/s causes reward to be frozen.")
print("  Since realistic max throughput >> 7.5 Mb/s, training signal is nearly constant.")
print()
print("BUG 3 (MODERATE): maddpg_train.py:277 back-calculates throughput using N_actual=40")
print("  but the reward was computed with N_hardcoded=5, causing 8x display overestimate.")
print()

# ---- Verify the 3.8 Mb/s number ----
# From the log: mean_episode_throughput_mbps ≈ 2.88
# From formula: throughput_mbps = reward * 6.0
# Observed reward ≈ 0.48 → throughput_mbps = 0.48 * 6.0 = 2.88 ✓
# Actual aggregate throughput = 0.48 * 7.5 = 3.6 Mb/s ≈ 3.8 Mb/s reported
print("=== Cross-check with observed numbers ===")
obs_r = 0.48
computed_oscar = obs_r * oscar_normaliser
actual_agg = obs_r * saturation_threshold_mbps
print(f"Observed reward: {obs_r}")
print(f"maddpg_train reported throughput: {computed_oscar:.2f} Mb/s  (overestimate by N_actual/N_hardcoded = {N_paper//denom_hardcoded_N}x)")
print(f"Actual aggregate throughput: {actual_agg:.2f} Mb/s")
print(f"Note: 3.8 Mb/s is likely the 'actual' from cw.cc end-of-run printf, not maddpg_train's inflated metric")
print()

# But the real_question: is 3.6-3.8 Mb/s the ACTUAL throughput, or is it additionally wrong?
# From the end-of-run printf in cw.cc:570:
#   res = final_total_rx * 1464 * 8 / 2^20 / simulationTime
# That's the true final throughput – not reward-based.
# So 3.8 Mb/s is the REAL FlowMonitor/packet-count aggregate.

print("=== Why 3.8 Mb/s instead of 38 Mb/s (10x deficit) ===")
print("Hypothesis: The N=40 reward normalisation bug prevents meaningful training,")
print("leaving actors near random/warm-start weights.")
print("At random actions (CW exponent ≈ U[4,10]), stations compete heavily,")
print("reducing effective throughput to ~10% of maximum.")

