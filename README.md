# Wi-Fi Contention-Window Reinforcement Learning

This repository contains ns-3 Gym experiments for selecting contention windows
in an 802.11 network with deep reinforcement learning.

The code currently provides:

- **Original OSCAR**: one centralized DDPG policy
- **Legacy MADDPG**: one shared actor with a mean-field critic
- **True MADDPG**: one actor per station with a centralized critic
- **CCOD/DDPG**: a separate DDPG implementation
- Fixed contention-window, BEB, and fairness baselines

## Requirements

- ns-3 with ns3-gym available
- This repository inside `scratch/linear-mesh`
- Python 3
- PyTorch
- NumPy
- W&B
- Gym-compatible dependencies required by ns3-gym

The ns-3 executable used by Python is named `linear-mesh`.

Authenticate with W&B before using the training scripts:

```bash
wandb login
```

Run commands from this directory:

```bash
cd ~/ns-allinone-3.29/ns-3.29/scratch/linear-mesh
```

## How to identify the implementations

### Original OSCAR

Original OSCAR is a single-agent DDPG experiment.

- Main runner: [`run_oscar_jsonl.py`](./run_oscar_jsonl.py)
- DDPG implementation: [`agents/our_ddpg/Our_DDPG.py`](./agents/our_ddpg/Our_DDPG.py)
- Replay buffer: [`agents/our_ddpg/utils.py`](./agents/our_ddpg/utils.py)
- Logging: [`agents/our_ddpg/loggers.py`](./agents/our_ddpg/loggers.py)

One policy sees the network state and produces one contention-window action.
The action is applied to every station.

```bash
python3 run_oscar_jsonl.py \
  --nWifi 40 \
  --simTime 15 \
  --episodes 12 \
  --seed 1 \
  --jsonl oscar_40_seed1.jsonl \
  --summary_json oscar_40_seed1.json
```

### Legacy MADDPG

Legacy MADDPG uses a shared actor and a mean-field critic.

- Main runner and training loop: [`maddpg_train.py`](./maddpg_train.py)
- Shared actor and mean-field critic:
  [`agents/maddpg/model.py`](./agents/maddpg/model.py)
- Joint replay buffer: [`agents/maddpg/utils.py`](./agents/maddpg/utils.py)

The same actor is used for every station. The critic approximates other
agents using mean observations and mean actions.

```bash
python3 maddpg_train.py \
  --nWifi 40 \
  --simTime 15 \
  --stepTime 0.01 \
  --episodes 12 \
  --seed 1 \
  --jsonl legacy40_seed1.jsonl \
  --summary_json legacy40_seed1.json
```

Do not add `--use_centralized_critic` for this variant.

### True MADDPG

True MADDPG uses CTDE: centralized training and decentralized execution.

- Main runner and training loop: [`maddpg_train.py`](./maddpg_train.py)
- `Actor`, `CentralizedCritic`, and actor initialization:
  [`agents/maddpg/model.py`](./agents/maddpg/model.py)
- Joint replay buffer: [`agents/maddpg/utils.py`](./agents/maddpg/utils.py)

There is one actor per station and one critic that receives all station
observations and actions.

```bash
python3 maddpg_train.py \
  --nWifi 40 \
  --simTime 15 \
  --stepTime 0.01 \
  --episodes 12 \
  --seed 1 \
  --use_centralized_critic \
  --jsonl true_maddpg40_seed1.jsonl \
  --summary_json true_maddpg40_seed1.json
```

### CCOD

CCOD is a separate DDPG implementation based on the RLinWiFi-style approach.
It is not the original OSCAR implementation.

- Training entry point: [`CCOD_train.py`](./CCOD_train.py)
- DDPG agent: [`agents/ddpg/agent.py`](./agents/ddpg/agent.py)
- Networks: [`agents/ddpg/model.py`](./agents/ddpg/model.py)
- Episode orchestration: [`agents/ddpg/teacher.py`](./agents/ddpg/teacher.py)

```bash
python3 CCOD_train.py
```

CCOD station count and episode settings are currently constants in
`CCOD_train.py`.

## Common environment code

The implementations use the same ns-3 environment contract:

- [`cw.cc`](./cw.cc): observations, actions, reward, telemetry, and CW updates
- [`scenario.h`](./scenario.h): Wi-Fi topology and traffic scenarios
- [`wrappers.py`](./wrappers.py): ns3-gym process and ZeroMQ interface

Each station observes its own frame-loss ratio:

```text
lost packets / transmitted packets
```

The observation passed to Python has shape `(N, 1)`. The action from Python
also has shape `(N, 1)`. The wrapper sends a flat N-element action to ns-3.

The policy action range is `[4, 10]`. ns-3 receives the corresponding
continuous value and computes:

```text
CW = 2^(policy_action)
```

In the MADDPG runners, Python therefore subtracts 4 before sending actions
because `cw.cc` adds 4 itself.

## Baselines

### Fixed contention-window sweep

```bash
python3 run_fixed_cw_sweep.py
```

This holds the same CW exponent on every station and sweeps values in the
`[4, 10]` policy range.

### Stage-one BEB baseline

```bash
python3 run_stage1_baselines.py \
  --seeds 1 2 3 \
  --sim-time 15.0 \
  --n-wifi 40 \
  --fixed-cw 16 32 64 128 256 512 1024 \
  --csv stage1_baselines.csv
```

### 802.11 BEB and CCOD comparison

```bash
python3 standard_test_and_ccod_train.py
```

The station counts and episode settings are currently constants in the script.

## Benchmark runners

Run legacy and true MADDPG with identical settings:

```bash
python3 run_step4_bench.py \
  --episodes 12 \
  --nWifi 40 \
  --seeds 1 2 3 4 5
```

`run_step4_bench.py` launches [`maddpg_train.py`](./maddpg_train.py) twice for
each seed: once without `--use_centralized_critic` and once with it.

Benchmark summaries and per-run logs are written under
`bench_results/` and are ignored by Git.

## Diagnostics and tests

Useful diagnostics include:

- [`run_action_diagnostics.py`](./run_action_diagnostics.py):
  compares OSCAR and MADDPG actions
- [`diag_action_stats.py`](./diag_action_stats.py):
  summarizes logged JSONL action statistics
- [`diag_throughput_formula.py`](./diag_throughput_formula.py):
  checks throughput calculations
- [`diag_ceiling_and_compare.py`](./diag_ceiling_and_compare.py):
  compares fixed-CW throughput ceilings
- [`diag/`](./diag): focused C++ and Python diagnostic scripts

Run the pure-Python MADDPG checks:

```bash
python3 tests/test_step1_verify.py
python3 tests/test_step4_centralized.py
python3 tests/test_measurement_fixes.py
python3 tests/test_fairness_jain.py
```

These tests do not require ns-3.

## Generated files

Generated experiment files are not intended to be committed.
The `.gitignore` excludes:

- JSON and JSONL experiment records
- CSV results
- PNG/SVG plots
- `.log` files
- `bench_results/`
- `results/`
- `wandb/`
- model checkpoints
- Python bytecode and caches

Keep source scripts, tests, documentation, and reusable configuration under
version control. Generated runs can be recreated with the commands above.

## Important naming caveat

[`maddpg_train.py`](./maddpg_train.py) is the MADDPG training entry point.

It supports two modes:

```text
No --use_centralized_critic
  -> legacy shared-actor MADDPG

With --use_centralized_critic
  -> true MADDPG with N actors and a centralized critic
```

The original OSCAR DDPG implementation is
[`run_oscar_jsonl.py`](./run_oscar_jsonl.py).

## Reference

```bibtex
@INPROCEEDINGS{10279663,
  author={Grasso, Christian and Raftopoulos, Raoul and Schembra, Giovanni},
  booktitle={ICC 2023 - IEEE International Conference on Communications},
  title={OSCAR: A Contention Window Optimization Approach Using Deep Reinforcement Learning},
  year={2023},
  volume={},
  number={},
  pages={459-465},
  doi={10.1109/ICC45041.2023.10279663}
}
```
