import numpy as np
import torch
import gym
import argparse
import os
import wandb
import tqdm
import subprocess
import time

from exceptions import AlreadyRunningException

from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path

class EnvWrapper:
    """Gym wrapper for the Dec-POMDP MADDPG architecture.

    The ns-3 backend now emits an N-dimensional observation vector
    (one localized frame-loss rate per station) and expects an
    N-dimensional continuous action vector (one CW parameter per station).

    This wrapper guarantees that:
      - reset()  returns  obs   of shape (N, 1)
      - step()   accepts  actions of shape (N, 1)  and returns
                  next_obs (N, 1), reward (scalar float),
                  done (bool), info (dict/str)

    Each parallel thread talks to its own ns-3 instance through a
    separate ZeroMQ port, so `no_threads` independent simulations
    can be run concurrently.
    """

    def __init__(self, no_threads, **params):
        self.params = params
        self.no_threads = no_threads
        self.n_agents = params.get('nWifi', 5)
        self.ports = [13968 + i + np.random.randint(40000) for i in range(no_threads)]
        self.commands = self._craft_commands(params)

        self.SCRIPT_RUNNING = False
        self.envs = []

        self.run()
        for port in self.ports:
            env = ns3env.Ns3Env(
                port=port,
                stepTime=params['envStepTime'],
                startSim=0,
                simSeed=params.get('seed', 1),
                simArgs=params,
                debug=False,
            )
            self.envs.append(env)

        self.SCRIPT_RUNNING = True

    def run(self):
        if self.SCRIPT_RUNNING:
            raise AlreadyRunningException("Script is already running")

        for cmd, port in zip(self.commands, self.ports):
            subprocess.Popen(['bash', '-c', cmd])
        self.SCRIPT_RUNNING = True

    def _craft_commands(self, params):
        try:
            waf_pwd = find_waf_path("./")
        except FileNotFoundError:
            import sys
            sys.path.append("../../")
            waf_pwd = find_waf_path("../../")

        command = f'{waf_pwd} --run "linear-mesh'
        for key, val in params.items():
            command += f" --{key}={val}"

        commands = []
        for p in self.ports:
            commands.append(command + f' --openGymPort={p}"')

        return commands

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all parallel environments.

        Returns:
            List[np.ndarray]:  One entry per thread.  Each entry has
                               shape ``(N, 1)`` where ``N = n_agents``.
        """
        obs = []
        for env in self.envs:
            raw = env.reset()
            # raw comes from ns3-gym as a flat iterable of length N.
            # Reshape to (N, 1) so every agent has its own scalar obs.
            obs.append(np.array(raw, dtype=np.float32).reshape(self.n_agents, 1))
        return obs

    def step(self, actions):
        """Execute one environment step across all threads.

        Args:
            actions: list/array of per-thread action arrays.
                     Each element should be shape ``(N, 1)`` (or ``(N,)``).
                     The array is flattened to a list and forwarded
                     directly to the ns3-gym ZeroMQ socket so that
                     the C++ side receives an N-dimensional action vector.

        Returns:
            next_obs  np.ndarray  shape ``(no_threads, N, 1)``
            reward    np.ndarray  shape ``(no_threads,)``   – shared global reward
            done      np.ndarray  shape ``(no_threads,)``
            info      np.ndarray  of objects
        """
        next_obs, reward, done, info = [], [], [], []

        for i, env in enumerate(self.envs):
            # Flatten (N,1) → list of N floats for the ZMQ protocol
            act = np.asarray(actions[i], dtype=np.float32).flatten().tolist()
            no, rew, dn, inf = env.step(act)
            # Reshape the raw N-element obs back to (N, 1)
            next_obs.append(
                np.array(no, dtype=np.float32).reshape(self.n_agents, 1)
            )
            reward.append(float(rew))
            done.append(dn)
            info.append(inf)

        return (
            np.array(next_obs),       # (threads, N, 1)
            np.array(reward),         # (threads,)
            np.array(done),           # (threads,)
            np.array(info),           # (threads,)
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        """Returns ``(no_threads, N, 1)``."""
        return (self.no_threads, self.n_agents, 1)

    @property
    def action_space(self):
        """Returns ``(no_threads, N, 1)``."""
        return (self.no_threads, self.n_agents, 1)

    # ------------------------------------------------------------------
    # lifecycle helpers
    # ------------------------------------------------------------------

    def close(self):
        time.sleep(1)
        for env in self.envs:
            env.close()
        self.SCRIPT_RUNNING = False

    def __getattr__(self, attr):
        for env in self.envs:
            env.attr()
