"""This file contains the implementation of DQN algorithm, along with a few improvements."""
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DuelingNetwork, QNetwork
from replay_buffer import PrioritizeReplayBuffer, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, state_size, action_size, config):
        """Initialize the agent.

        Args:
            state_size (int): Size of the state.
            action_size (int): Size of the actions.
            config (DefaultConfig): Configuration object. See config.py for details.
        """
        random.seed(config.SEED)

        self.action_size = action_size

        # Models
        model_type = DuelingNetwork if config.DUELING else QNetwork
        self.model = model_type(state_size, action_size, config.SEED).to(device)
        self.target_model = model_type(state_size, action_size, config.SEED).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Replay buffer
        replay_buffer_type = PrioritizeReplayBuffer if config.PER else ReplayBuffer
        self.memory = replay_buffer_type(
            config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED
        )

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR)

        # Epsilon for esp-greedy policy
        self.eps = config.EPS_START

        self.config = config

        # Count step to update only every k steps
        self.t_step = 0

    def step(self):
        """Run a step of gradient descent."""
        states, actions, rewards, next_states, dones = self._draw_minibatch()

        # Select next action
        if self.config.DOUBLE:
            actions_max = self.model(next_states).argmax(1, keepdim=True)
        else:
            actions_max = self.target_model(next_states).argmax(1, keepdim=True)

        # Compute target
        target_Q = self.target_model(next_states).gather(1, actions_max).detach()
        target = rewards + self.config.GAMMA * (1 - dones) * target_Q

        # Compute Q value
        Q = self.model(states).gather(1, actions)

        # Get TD error
        td_error = target - Q

        # Compute loss and update priorities if needed
        if self.config.PER:
            weights = torch.Tensor(self.memory.importance_sampling()).unsqueeze(1)
            self.memory.update_priorities(td_error.detach().cpu().numpy())
            loss = torch.mean(weights * td_error ** 2)
        else:
            loss = torch.mean(td_error ** 2)

        # Optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target network
        self._soft_update()

    def train(self, env, num_episode, max_t=300):
        """Train the agent.

        Args:
            env: Environment in which the agent is trained.
            num_episode (int): Number of training episodes.
            max_t (int): Max number of time steps per episode.

        Returns:
            List of the episodes running means (last 100 episodes).
        """
        scores = []
        scores_window = deque(maxlen=100)

        for episode in range(num_episode):

            # Init state and episode score
            state = env.reset(train_mode=True)
            score = 0

            # Run episode
            for t in range(max_t):

                # Select and run action
                action = self._eps_greedy(state)
                next_state, reward, done = env.step(action)

                # Store in replay buffer
                self.memory.add(state, action, reward, next_state, done)

                # Update time step
                self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY

                # Optimisation step if UPDATE_EVERY and enough examples in memory
                if self.t_step == 0 and len(self.memory) > self.config.BATCH_SIZE:
                    self.step()

                # Update state and scores
                state = next_state
                score += reward

                if done:
                    break

            # Keep track of running mean
            scores_window.append(score)

            # Append current mean to scores list
            scores.append(np.mean(scores_window))

            # Update eps
            self.eps = max(self.config.EPS_END, self.config.EPS_DECAY * self.eps)

            # Logging
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_window)
                ),
                end="",
            )
            if (episode + 1) % 100 == 0:
                print(
                    "\rEpisode {}\tAverage Score: {:.2f}".format(
                        episode, np.mean(scores_window)
                    )
                )

        return scores
    
    def save_model(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def _draw_minibatch(self):
        """Draw a minibatch in the replay buffer."""
        states, actions, rewards, next_states, done = zip(*self.memory.sample())

        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).long().unsqueeze(1).to(device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(device)
        next_states = torch.Tensor(next_states).to(device)
        done = torch.Tensor(done).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, done

    def _eps_greedy(self, state):
        """Epsilon Greedy policy."""
        if random.random() > self.eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.model.eval()

            with torch.no_grad():
                action_values = self.model(state)

            self.model.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _soft_update(self):
        for target_param, local_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.config.TAU * local_param.data
                + (1.0 - self.config.TAU) * target_param.data
            )
