
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9995,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))

        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, terminated):
        best_next_value = np.max(self.q_table[next_state])

        target = reward + self.gamma * best_next_value * (
            1 - int(terminated)
        )

        self.q_table[state, action] += self.alpha * (
            target - self.q_table[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

