"""
diag/exact_math_trace.py
Trace the exact mathematical transformations between actual packet count,
reward in cw.cc, and reported throughput in maddpg_train.py.
"""
N = 40
stepTime = 0.01  # seconds
pkt_payload = 1500 - 20 - 8 - 8  # 1464 bytes

# Suppose the network achieves 38.0 Mb/s (OSCAR paper target)
true_throughput_mbps = 38.0

# In 1 step of 0.01s:
megabits_in_step = true_throughput_mbps * stepTime  # 0.3800 Mb
# Number of packets received in that step:
res = (megabits_in_step * 1024 * 1024) / (pkt_payload * 8.0)
print(f"Target throughput: {true_throughput_mbps} Mb/s")
print(f"Packets per 10ms step (res): {res:.2f}")

# cw.cc line 212:
# float reward = res * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024 / (5 * 150 * envStepTime) * 10;
cw_cc_megabits = res * pkt_payload * 8.0 / 1024 / 1024
cw_cc_reward = cw_cc_megabits / (5 * 150 * stepTime) * 10.0
print(f"cw.cc calculated reward: {cw_cc_reward:.4f}")

# maddpg_train.py line 277:
# throughput_mbps = reward * (N * 150.0 * args.stepTime) / 10.0
reported_tput = cw_cc_reward * (N * 150.0 * stepTime) / 10.0
print(f"maddpg_train reported throughput: {reported_tput:.4f} Mb/s")
print(f"Ratio (reported / true): {reported_tput / true_throughput_mbps:.4f}")
print(f"Under-reporting factor: {true_throughput_mbps / reported_tput:.2f}x")
print()

# Now trace in reverse: what true throughput corresponds to the reported ~2.88 - 3.8 Mb/s?
for rep in [2.50, 2.88, 3.80]:
    # rep = reward * (40 * 150 * 0.01) / 10 = reward * 6.0
    r = rep / 6.0
    # r = Mb / 7.5 * 10 = Mb / 0.75 -> Mb = r * 0.75
    mb = r * 0.75
    # true_tput = mb / stepTime = mb * 100
    true_tput = mb / stepTime
    print(f"Reported {rep:.2f} Mb/s -> cw.cc reward = {r:.4f} -> True physical throughput = {true_tput:.2f} Mb/s")

