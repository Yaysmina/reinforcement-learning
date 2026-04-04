import numpy as np

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, alpha: float = 0.2, 
                 gamma: float = 0.9, epsilon: float = 0.2, epsilon_decay: float = 0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state: int) -> int:
        # If exploration, choose a random action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # If exploitation, choose the action with the highest Q-value
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state: int, action: int, reward: float, new_state: int, is_terminal: bool) -> None:
        old_value = self.q_table[state, action]
        
        estimated_future_reward = 0 if is_terminal else np.max(self.q_table[new_state])

        # Temporal Difference Learning -> TD(0)
        new_value = old_value + self.alpha * (reward + self.gamma * estimated_future_reward - old_value)
        self.q_table[state, action] = new_value

    def get_value_table(self, grid_size: int) -> np.ndarray:
        one_dim_table = np.max(self.q_table, axis=1)
        return one_dim_table.reshape((grid_size, grid_size))
    
    def decay_epsilon(self) -> None:
        self.epsilon *= (1 - self.epsilon_decay)