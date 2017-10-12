import numpy as np
import cntk as C
from cntk.logging import TensorBoardProgressWriter


class CategoricalAgent:
    def __init__(self, state_shape, action_count, model_func, vmin, vmax, n,
                 gamma=0.99, lr=0.00025, mm=0.95, use_tensorboard=False):
        """
        Creates a new agent that learns using Categorical DQN as described in
        "A Distributional Perspective on Reinforcement Learning"
        :param state_shape: The shape of each input shape e.g. (4 x 84 x 84) for Atari
        :param action_count: The number of actions e.g. 14
        :param model_func: The model to train
        :param vmin: Minimum value of return distribution
        :param vmax: Maximum value of return distribution
        :param n: Number of support atoms
        :param gamma: Discount factor for Bellman update
        :param lr: The learning rate for Adam SGD
        :param mm: The momentum for Adam SGD
        """
        self.state_shape = state_shape
        self.action_count = action_count
        self.gamma = gamma
        self.learning_rate = lr
        self.momentum = mm

        # Distribution parameters
        self.vmin = vmin
        self.vmax = vmax
        self.n = n
        self.dz = (vmax - vmin) / (n - 1)

        # Support atoms
        self.z = np.linspace(vmin, vmax, n, dtype=np.float32)

        # Model input and output
        self.state_var = C.input_variable(self.state_shape, name='state')
        self.action_return_dist = C.input_variable((self.action_count, n), name='ar_dist')

        # Model output assigns a probability to each support atom for each action
        self.raw = model_func(self.state_var)
        self.model = C.softmax(self.raw, axis=1)

        # Adam-based SGD with cross-entropy loss
        loss = C.cross_entropy_with_softmax(self.raw, self.action_return_dist, axis=1, name='loss')
        lr_schedule = C.learning_rate_schedule(self.learning_rate, C.UnitType.sample)
        mom_schedule = C.momentum_schedule(self.momentum)
        vm_schedule = C.momentum_schedule(0.999)
        learner = C.adam(self.raw.parameters, lr_schedule, mom_schedule, variance_momentum=vm_schedule)

        if use_tensorboard:
            self.writer = TensorBoardProgressWriter(log_dir='metrics', model=self.model)
        else:
            self.writer = None

        self.trainer = C.Trainer(self.raw, (loss, None), [learner], self.writer)

        # Create target network as copy of online network
        self.target_model = None
        self.update_target()

    def update_target(self):
        """
        Create a fixed copy of the online network
        """
        self.target_model = self.model.clone(C.CloneMethod.freeze)

    def act(self, state, epsilon):
        """
        Return an action to take based on the epsilon-greedy method
        :param state: The current state
        :param epsilon: Value in range [0, 1].
        :return: Next action to execute
        """
        if np.random.uniform(0, 1) < epsilon:
            # Take random action (explore)
            return np.random.choice(self.action_count)
        else:
            # Take action that maximizes return (exploit knowledge)
            action_returns = np.dot(self.model.eval(state)[0], self.z)
            return np.argmax(action_returns)

    def train(self, states, actions, rewards, post_states, terminals):
        """
        Updates the model using the given minibatch date
        :param states: Batch of start states in each transition
        :param actions: Batch of actions taken in each transition
        :param rewards: Batch of rewards received for each action
        :param post_states: Batch of resulting states for each action
        :param terminals: Indicates if each post state is a terminal state
        """
        batch_size = len(states)

        # P[k, a, i] is the probability of atom z_i when action a is taken in the next state (for the kth sample)
        P = self.model.eval(post_states)

        # Q[k, a] is the value of action a (for the kth sample)
        Q = np.dot(P, self.z)

        # A_[k] is the optimal action (for the kth sample)
        A_ = np.argmax(Q, axis=1)

        # Target vector
        M = np.zeros((batch_size, self.action_count, self.n), dtype=np.float32)

        # Compute projection onto the support (for terminal states, just reward)
        Tz = np.repeat(rewards.reshape(-1, 1), self.n, axis=1) + np.dot(self.gamma * (1.0 - terminals).reshape(-1, 1),
                                                                        self.z.reshape(1, -1))

        # TODO: Verify correctnes
        # Clipping to endpoints like described in paper causes probabilities to disappear (when B = L = U).
        # To avoid this, I shift the end points to ensure that L and U are not both equal to B
        Tz = np.clip(Tz, self.vmin + 0.01, self.vmax - 0.01)

        B = (Tz - self.vmin) / self.dz
        L = np.floor(B).astype(np.int32)
        U = np.ceil(B).astype(np.int32)

        # Distribute probability
        for i in range(batch_size):
            for j in range(self.n):
                M[i, A_[i], L[i, j]] += P[i, A_[i], j] * (U[i, j] - B[i, j])
                M[i, A_[i], U[i, j]] += P[i, A_[i], j] * (B[i, j] - L[i, j])

        # Train using computed targets
        self.trainer.train_minibatch({self.state_var: states, self.action_return_dist: M})

    def checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def save_model(self, filename):
        self.model.save(filename)
