# Project 1: Navigation

The environment is described as Markov Decision Process $(S, A, R, p, \gamma)$, which dynamics is unknown.
The goal is to approximate the optimal policy $\pi^*$ maximizing the upcoming cumulative reward in every state.

A value based algorithm has been used, which seeks to approximate the optimal $q$ value $q_*$ which is defined as:
$$
q_*(s, a) = \max_{\pi} \mathbb{E}_{\pi} (\sum_{t} \gamma^t r_t | s_0=s, a_0=a),  \forall (a, s) \in A \times S
$$

Given $q_*$, one can derive $\pi_*$ to solve the problem:
$$
\pi_*(s) = \argmax_a q_*(s, a), \forall s \in S
$$

## The algorithm

### DQN

The algorithm used to solve the environment is [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), where the $q$ value is approximated by a neural network.

Let $Q$ be this neural network, parameterized by $\theta$.
The loss function is defined as the mean squared error in the Bellman optimality equation:
$$
\mathcal{L}(\theta) = \mathbb{E}_{s, a, r, s'} \big[((r + \gamma \max_{a'} Q(s', a' | \theta^-)) - Q(s, a| \theta))^2\big]
$$

$\theta^-$ is a second set of parameter, defining a target network, in order for the target $r + \gamma \max_{a'} Q(s', a' | \theta^-)$ to not depend on the parameters $\theta$.

This loss is minimized by a stochastic gradient descent algorithm.
At each step, the gradient is approximated from a sampled minibatch of experiences $\{(s, a, r, s', d)_i\}$.
The experiences in the minibatch are supposed to be i.i.d in order for the sampled gradient to be a good approximation.
This is not the case if the last experiences are used because they are strongly correlated.
Thus, experiences are stored in a replay buffer, which capacity is big compared to the batch size.
Each minibatch is then drawn uniformly from this replay buffer.

### Improvements

Three improvements have been implemented for this project:
* [Double DQN](https://arxiv.org/abs/1509.06461):
    Instead of using the target network to chose the next action and estimate its value in the target, use $Q_\theta$ to chose the action, and $Q_{\theta^-}$ to estimate its value.
* [Dueling DQN](https://arxiv.org/abs/1511.06581):
    Define a network with two streams: one to evaluate the state value function and one for the advantage of each action. They are then merged to result in the $Q$ value.
* [Prioritized experience replay](https://arxiv.org/abs/1511.05952)
    Smarter way of sampling in the replay buffer. Instead of uniform probability, assign a priority based on the TD-error to each sample. The proportional version has been implemented here.

## Implementation

The network architecture is composed of 3 fully connected layers, with respectively 64, 64 and 32 neurons each.
The optimizer is Adam with a learning rate of $5.10^{-5}$.
The discount factor $\gamma$ is equal to $0.99$. 
The agent follows an $\epsilon$-greedy policy, with epsilon gradually decreasing from $1$ to $0.1$.
The target parameters $\theta^-$ are updated towards $\theta$ with a momentum $\tau$ of $10^-3$.

The double DQN uses these same hyperparameters and architecture.

On top of this double DQN, a dueling architecture has been developped.
After a common layer of 64 neurons, the value and advantage streams both contain 2 layers of 64 and 32 neurons.

On top of this dueling double DQN, a prioritize replay buffer has been developped. It is based on the proportional variant defined on the paper, and thus rely on a sum tree for an effective implementation.
The $\alpha$ parameter has been fixed to 0.6 while the $\beta$ gradually increase from 0.4 to 1.
The learning rate is divided by 4, as suggested in the paper.

## Results

The 4 configuration have been tested: DQN, Double DQN, Dueling Double DQN, Dueling Double DQN Prioritize.
The results (plot and number of episodes to solve) can be seen in the `Navigation.ipynb` notebook.

Unfortunately, vanilla DQN seems to slightly perform better than others, solving the environment in 270 episodes.
This may be due to hyperparameters being tuned on vanilla and then reused for other variant, or to the relative simplicity of the problem.
Indeed, these improvements have been developped for deep neural network, while this environment can be solved with a few fully connected layers only.

The training were carried out for 500 episodes, and the weights for each configuration are provided in the repository.

## What next ?
To improve the performances of the model, some hyperparameters tuning may be done, especially on the improved versions.
The next step would be to implement other improvements to aim for rainbow DQN. 
The first one I want to try when I have time is multi-step return.
