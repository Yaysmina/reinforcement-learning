import time
import torch
from environment import GridMuckEnvV2
from model import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    """
    Agent class
    """
    def __init__(self, hyper_parameters_set: str = "version_2"):
        """
        Initializes the agent
        """
        with open("version_3/hyper_parameters.yml", "r") as file:
            all_hyper_parameters = yaml.safe_load(file)
            hyper_parameters = all_hyper_parameters[hyper_parameters_set]

        # Extract the hyper parameters
        self.replay_memory_size = hyper_parameters["replay_memory_size"]
        self.mini_batch_size    = hyper_parameters["mini_batch_size"]
        self.epsilon_init       = hyper_parameters["epsilon_init"]
        self.epsilon_decay      = hyper_parameters["epsilon_decay"]
        self.epsilon_min        = hyper_parameters["epsilon_min"]

    def run(self, is_training: bool = True, render: bool = False):
        """
        Runs the agent
        """
        # Initialize the environment
        env = GridMuckEnvV2(size=7, render_mode="human" if render else None)
        obs, info = env.reset()

        # Initialize the model
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy = DQN(num_states, num_actions).to(device)

        # Keep track of rewards
        rewards_per_episode = []
        epsilon_history = []

        if is_training:
            # Initialize the replay memory
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

        for episode in itertools.count():
            # Initialize the environment
            state, _ = env.reset()
            state = self._to_tensor_float(state)
            terminated = False
            truncated = False
            episode_reward = 0.0

            # Play the game once
            while not (terminated or truncated):
                
                # Epsilon-greedy policy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = self._to_tensor_long(action)
                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(0)).squeeze(0).argmax()
 
                # Take step
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate reward
                episode_reward += reward

                # Convert state and reward to tensor
                new_state = self._to_tensor_float(new_state)
                reward = self._to_tensor_float(reward)

                # Store transition in replay memory
                if is_training:
                    memory.append((state, action, new_state, reward, terminated or truncated))

                # Move to new state
                state = new_state
                # time.sleep(0.1)

            # Add reward to list
            rewards_per_episode.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward}")

            # Update epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)
    
    
    def _to_tensor_float(self, value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    def _to_tensor_long(self, value):
        return torch.tensor(value, dtype=torch.long, device=device)


if __name__ == "__main__":
    agent = Agent("version_2")
    agent.run(is_training=True, render=True)