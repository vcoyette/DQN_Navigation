import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """DQN Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialise model."""
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, state):
        """Forward pass, returns Q values for each action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialise model."""
        super(DuelingNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)

        self.fc2v = nn.Linear(64, 64)
        self.fc3v = nn.Linear(64, 32)
        self.fc4v = nn.Linear(32, 1)

        self.fc2a = nn.Linear(64, 64)
        self.fc3a = nn.Linear(64, 32)
        self.fc4a = nn.Linear(32, action_size)

    def forward(self, state):
        """Forward pass, returns Q values for each action."""
        common = F.relu(self.fc1(state))
        
        value = F.relu(self.fc2v(common))
        value = F.relu(self.fc3v(value))
        value = self.fc4v(value)

        advantage = F.relu(self.fc2a(common))
        advantage = F.relu(self.fc3a(advantage))
        advantage = self.fc4a(advantage)

        return value + advantage - advantage.mean(1, keepdim=True)
