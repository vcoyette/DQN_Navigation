from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from config import DDQNConfig, DuelingDDQNConfig, PERDDQNConfig
from dqn import DQN
from environment import UnityGymAdapter


def train():
    """Run training."""
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    env = UnityGymAdapter(env)

    config = PERDDQNConfig()

    agent = DQN(env.state_size, env.action_size, config)

    scores = agent.train(env, 1800)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    train()
