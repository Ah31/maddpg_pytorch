import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

# if bn is done in here then both actor and critic loss are increasing and critic loss is increasing much faster!
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # torch.manual_seed(seed)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(out_features=fc1_units, in_features=state_size)
        self.fc2 = nn.Linear(out_features=fc2_units, in_features=fc1_units)
        self.fc3 = nn.Linear(out_features=action_size, in_features=fc2_units)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer

        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(out_features=fcs1_units, in_features = state_size*3)
        self.fc2 = nn.Linear(out_features=fc2_units, in_features=fcs1_units+(action_size*3))
        self.fc3 = nn.Linear(out_features=1, in_features=fc2_units)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, corresponding actions) pairs -> Q-values.
        And the overall goal is to maximise the Q value for these states."""

        xs = F.relu(self.fcs1(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
