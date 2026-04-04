import numpy as np
from game import Map

rng = np.random.default_rng()

class Q_learning_agent:

    def __init__(self, map, alpha = 0.2, gamma = 0.9, epsilon = 0.2, epsilon_decay = 0.05):
        self.map = map
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay # Exploration rate decay after each episode

        self.q_table = np.zeros((map.size * map.size, 4))

    def get_action(self, state):
        # If exploration, return a random action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(self.q_table[state])
    
    def take_step(self, state, testing = False):
        action = self.get_action(state)
        reward = self.map.take_action(action)
        new_state = self.map.get_state()
        is_terminal = self.map.is_terminal_state() 
        
        if not testing:
            self.update_q_table(state, action, reward, new_state, is_terminal)
    
    def update_q_table(self, state, action, reward, new_state, is_terminal):
        old_S_A_value = self.q_table[state, action]
        
        if is_terminal:
            estimated_future_R = 0
        else:
            estimated_future_R = np.max(self.q_table[new_state])

        # Q-learning formula
        new_S_A_value = old_S_A_value + self.alpha * (reward + self.gamma * estimated_future_R - old_S_A_value)
        self.q_table[state, action] = new_S_A_value

    def get_value_table(self):
        one_dim_table = np.max(self.q_table, axis = 1)
        two_dim_table = one_dim_table.reshape(self.map.size, self.map.size)
        return two_dim_table
    
    def update_epsilon(self):
        self.epsilon *= (1 - self.epsilon_decay)