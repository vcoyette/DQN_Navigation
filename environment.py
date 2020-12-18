"""Define environment related functions."""


class UnityGymAdapter:
    """Adapt Unity ML agent api to be closer to gym.

    This class aims at abstracting away elements such as brain names and env_info
    in Unity Environments.
    This result in cleaner calls to reset and step functions, similar to gym api.
    This is is not a full adapter though, only the functions needed in this project
    are adapted.
    """

    def __init__(self, env):
        """Initialize the environment.

        Args:
            env (UnityEnvironment): The Unity environment to be adapted.
        """
        self.env = env
        self.brain_name = self.env.brain_names[0]
        self.action_size = env.brains[self.brain_name].vector_action_space_size
        self.state_size = len(self.reset(False))

    def reset(self, train_mode):
        """Reset the environment.

        Args:
            train_mode (bool): True to use training mode, False for test.
        """
        env_info = self.env.reset(train_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, action):
        """Run a step of the environment.

        Args:
            action (array_like): The action to be performed.
        """
        env_info = self.env.step(action)[self.brain_name]

        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        return next_state, reward, done

    def close(self):
        """Close the environment."""
        self.env.close()