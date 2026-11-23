"""
diag/training_curve_plot.py
Parse the latest wandb output.log and produce training-curve PNGs in diag/.
Run from the scratch/linear-mesh directory.
"""
import os, re, sys
import numpy as np

# Try to import matplotlib; fail gracefully if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available – printing tables only")

LOGFILE = os.path.join(os.path.dirname(__file__), "..",
                       "wandb/run-20261005_195735-64byxwec/files/output.log")

DIAG_DIR = os.path.dirname(__file__)

# ── Parse tqdm / tpt lines ──────────────────────────────────────────────────
# Format (tqdm postfix):
#   reward=0.42, q_loss=0.008, pi_loss=-0.447, tput=2.50Mb/s
# Episode lines:
#   Episode N | Reward: ... | Mean Throughput: X.XX Mb/s

reward_hist    = []
q_loss_hist    = []
pi_loss_hist   = []
tput_hist      = []
ep_rewards     = []
ep_tput        = []

pat_step   = re.compile(r"reward=([\d.]+).*?q_loss=([-\d.]+).*?pi_loss=([-\d.]+).*?tput=([\d.]+)Mb")
pat_ep     = re.compile(r"Episode\s+(\d+)\s*\|\s*Reward:\s*([\d.]+)\s*\|\s*Mean Throughput:\s*([\d.]+)")

with open(LOGFILE) as f:
    for line in f:
        m = pat_step.search(line)
        if m:
            reward_hist.append(float(m.group(1)))
            q_loss_hist.append(float(m.group(2)))
            pi_loss_hist.append(float(m.group(3)))
            tput_hist.append(float(m.group(4)))
        m2 = pat_ep.search(line)
        if m2:
            ep_rewards.append(float(m2.group(2)))
            ep_tput.append(float(m2.group(3)))

print(f"Parsed {len(reward_hist)} step rows, {len(ep_rewards)} episodes")
if ep_tput:
    print("Episode mean throughputs (Mb/s):", [f"{x:.2f}" for x in ep_tput])
    print("Episode total rewards:          ", [f"{x:.1f}" for x in ep_rewards])

if not HAS_MPL:
    sys.exit(0)

steps = np.arange(len(reward_hist))

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle("maddpg_train.py Training Curves (run-20261005)", fontsize=13)

axes[0].plot(steps, reward_hist, lw=0.6, color="steelblue")
axes[0].set_ylabel("Global Reward")
axes[0].set_title("Global Reward per step  (range ≈ 0.2–0.5 across all episodes → no learning)")

axes[1].plot(steps, q_loss_hist, lw=0.6, color="darkorange")
axes[1].set_ylabel("Critic Loss (MSE)")
axes[1].set_title("Critic Loss  (low ≈ 0.003–0.01 → critic fits the flat reward well)")

axes[2].plot(steps, pi_loss_hist, lw=0.6, color="green")
axes[2].set_ylabel("Actor Loss (-Q)")
axes[2].set_title("Actor Loss  (constant ≈ -0.45 → actor not improving Q)")

axes[3].plot(steps, tput_hist, lw=0.6, color="crimson")
axes[3].axhline(38.0, color="k", lw=1.5, ls="--", label="OSCAR paper target (38 Mb/s)")
axes[3].set_ylabel("Throughput (Mb/s)")
axes[3].set_title("Reported Throughput Mb/s  (≈2.5–3.0 vs 38 target)")
axes[3].legend()
axes[3].set_xlabel("Training Step")

plt.tight_layout()
outpath = os.path.join(DIAG_DIR, "training_curves.png")
plt.savefig(outpath, dpi=120)
print(f"Saved: {outpath}")
plt.close()

# Episode bar chart
if ep_tput:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(range(len(ep_tput)), ep_tput, color="steelblue")
    ax2.axhline(38.0, color="red", lw=2, ls="--", label="Paper target 38 Mb/s")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Throughput (Mb/s)")
    ax2.set_title("Per-Episode Mean Throughput")
    ax2.legend()
    ep_path = os.path.join(DIAG_DIR, "episode_throughput.png")
    plt.savefig(ep_path, dpi=120)
    print(f"Saved: {ep_path}")
    plt.close()

