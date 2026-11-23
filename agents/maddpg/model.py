"""MADDPG Actor and Centralized-Critic networks for Dec-POMDP CW control.

Architecture
------------
* **Actor** (Decentralized):
    Input  → local observation  o_i   (dim = 1)
    Output → continuous action  a_i   (dim = 1, range [4, 10])

    All N Actor instances are created from the same base template weights
    (homogeneous initialisation) to accelerate cooperative convergence.

* **MeanFieldCritic**:
    Input  → local obs, local action, mean obs of others, mean action of others (dim = 4)
    Output → scalar Q_i(o_i, a_i, mean_o, mean_a)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    """Uniform init bounds based on fan-in."""
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


# ======================================================================
# Decentralised Actor
# ======================================================================

class Actor(nn.Module):
    """Decentralised actor for a single agent.

    Maps a scalar local observation ``o_i`` to a scalar continuous
    action ``a_i`` in the range ``[4, 10]``.

    The action range is enforced by a sigmoid output scaled and
    shifted to ``[4, 10]``.
    """

    ACTION_LOW = 4.0
    ACTION_HIGH = 10.0

    def __init__(self, obs_dim=1, act_dim=1, fc1_units=64,
                 fc2_units=64, seed=-1):
        """
        Args:
            obs_dim:    dimension of local observation (default 1)
            act_dim:    dimension of local action      (default 1)
            fc1_units:  hidden-layer 1 width
            fc2_units:  hidden-layer 2 width
            seed:       random seed (-1 → no manual seed)
        """
        super(Actor, self).__init__()
        if seed != -1:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(obs_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, act_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        """
        Args:
            obs: tensor of shape ``(batch, 1)``

        Returns:
            action: tensor of shape ``(batch, 1)`` in ``[4, 10]``
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # Sigmoid → [0, 1] then scale to [4, 10]
        x = torch.sigmoid(self.fc3(x))
        return self.ACTION_LOW + (self.ACTION_HIGH - self.ACTION_LOW) * x


# ======================================================================
# Helper: create N homogeneous actors
# ======================================================================

def create_homogeneous_actors(n_agents, obs_dim=1, act_dim=1,
                              fc1_units=64, fc2_units=64, seed=42):
    """Instantiate ``n_agents`` Actor networks sharing the same initial weights.

    A single *template* Actor is built first (with random weights
    θ_base). Every subsequent Actor is created by deep-copying that
    template, guaranteeing identical starting parameters across all
    agents.

    Args:
        n_agents:   number of agents N
        obs_dim:    local observation dimension
        act_dim:    local action dimension
        fc1_units:  first hidden-layer width
        fc2_units:  second hidden-layer width
        seed:       random seed for the template

    Returns:
        list[Actor] of length ``n_agents``, all on ``device``
    """
    template = Actor(obs_dim, act_dim, fc1_units, fc2_units, seed=seed).to(device)
    actors = []
    for _ in range(n_agents):
        actor_copy = Actor(obs_dim, act_dim, fc1_units, fc2_units, seed=-1).to(device)
        actor_copy.load_state_dict(copy.deepcopy(template.state_dict()))
        actors.append(actor_copy)
    return actors


# ======================================================================
# Centralized Critic (True MADDPG)
# ======================================================================

class CentralizedCritic(nn.Module):
    """True MADDPG centralized critic.

    Input: joint obs (batch, N) + joint actions (batch, N) -> dim = 2N
    Output: scalar Q(o, a)
    """
    def __init__(self, n_agents, obs_dim=1, act_dim=1,
                 fc1_units=128, fc2_units=128, seed=42):
        super().__init__()
        if seed != -1:
            torch.manual_seed(seed)
        input_dim = n_agents * (obs_dim + act_dim)   # 2N
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, joint_obs, joint_actions):
        # joint_obs:     (batch, N)
        # joint_actions: (batch, N)
        x = torch.cat([joint_obs, joint_actions], dim=1)  # (batch, 2N)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ======================================================================
# Mean Field Critic (Legacy - kept for A/B comparison)
# ======================================================================

class MeanFieldCritic(nn.Module):
    """Mean-Field Q-value critic for scalable MADDPG.

    Input:  local obs, local action, mean other obs, mean other action (dim=4)
    Output: scalar Q_i
    """

    def __init__(self, obs_dim=1, act_dim=1, fc1_units=128, fc2_units=64, seed=42):
        super(MeanFieldCritic, self).__init__()
        if seed != -1:
            torch.manual_seed(seed)

        input_dim = (obs_dim + act_dim) * 2  # 4

        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, local_o, local_a, mean_o, mean_a):
        x = torch.cat([local_o, local_a, mean_o, mean_a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
