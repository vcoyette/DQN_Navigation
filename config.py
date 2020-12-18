"""This file contains configuration objects."""


class DefaultDQNConfig:
    """Default DQN configuration."""

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network
    EPS_START = 1.0  # Initial epsilon value
    EPS_END = 0.01  # Smallest epsilon value
    EPS_DECAY = 0.95  # Epsilon decay
    DOUBLE = False  # Use double DQN
    DUELING = False  # Use Dueling DQN
    PER = False  # Use Prioritize Experience Replay
    SEED = 0  # Fix seed


class DDQNConfig(DefaultDQNConfig):
    """Double DQN configuration."""

    DOUBLE = True


class DuelingDDQNConfig(DDQNConfig):
    """Dueling Double DQN COnfiguration."""

    DUELING = True


class DuelingDDQNConfigPER(DuelingDDQNConfig):
    """Prioritize Experience Replay Dueling Double DQN config."""

    LR = 5e-4 / 4
    PER = True
