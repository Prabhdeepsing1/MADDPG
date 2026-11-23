"""Multi-Agent replay buffer for MADDPG.

Stores joint transition tuples::

    (o_n, a_n, r_{n+1}, o_{n+1}, done)

where ``o_n`` and ``a_n`` are arrays of length N (one entry per agent)
and ``r_{n+1}`` is the scalar shared global reward.
"""

import numpy as np
import torch


class MADDPGReplayBuffer:
    """Fixed-size ring buffer for joint (multi-agent) transitions.

    Internal storage uses pre-allocated NumPy arrays for cache-friendly
    random access. The ``sample`` method returns PyTorch tensors ready
    for mini-batch SGD.

    Shapes (stored)
    ---------------
    joint_obs       (max_size, n_agents)
    joint_actions   (max_size, n_agents)
    rewards         (max_size, n_agents)
    next_joint_obs  (max_size, n_agents)
    not_done        (max_size, 1)
    """

    def __init__(self, n_agents, obs_dim=1, act_dim=1,
                 max_size=int(1e6)):
        """
        Args:
            n_agents:  number of agents N
            obs_dim:   per-agent observation dimensionality (1)
            act_dim:   per-agent action dimensionality      (1)
            max_size:  maximum number of transitions stored
        """
        self.max_size = int(max_size)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0

        total_obs_dim = n_agents * obs_dim     # N
        total_act_dim = n_agents * act_dim     # N

        self.joint_obs = np.zeros((self.max_size, total_obs_dim),
                                  dtype=np.float32)
        self.joint_actions = np.zeros((self.max_size, total_act_dim),
                                      dtype=np.float32)
        self.rewards = np.zeros((self.max_size, n_agents), dtype=np.float32)
        self.next_joint_obs = np.zeros((self.max_size, total_obs_dim),
                                       dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(self, joint_obs, joint_actions, reward, next_joint_obs, done):
        """Append a single joint transition.

        Args:
            joint_obs:       array-like of length N  (flat or (N,1))
            joint_actions:   array-like of length N
            reward:          array-like of length N (shaped rewards)
            next_joint_obs:  array-like of length N
            done:            bool or 0/1
        """
        self.joint_obs[self.ptr] = np.asarray(joint_obs,
                                              dtype=np.float32).flatten()
        self.joint_actions[self.ptr] = np.asarray(joint_actions,
                                                   dtype=np.float32).flatten()
        self.rewards[self.ptr] = np.asarray(reward, dtype=np.float32).flatten()
        self.next_joint_obs[self.ptr] = np.asarray(next_joint_obs,
                                                    dtype=np.float32).flatten()
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size):
        """Uniformly sample a mini-batch of joint transitions.

        Args:
            batch_size (int): number of transitions to sample.

        Returns:
            Tuple of five ``torch.FloatTensor`` on ``self.device``:

            * ``joint_obs``       – ``(batch, N)``
            * ``joint_actions``   – ``(batch, N)``
            * ``rewards``         – ``(batch, N)``
            * ``next_joint_obs``  – ``(batch, N)``
            * ``not_done``        – ``(batch, 1)``
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.joint_obs[ind]).to(self.device),
            torch.FloatTensor(self.joint_actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_joint_obs[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self):
        return self.size
