import time
import torch
from environment import GridMuckEnvV2
from model import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import numpy as np

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

        # Hyper parameters
        self.replay_memory_size = hyper_parameters["replay_memory_size"]
        self.mini_batch_size    = hyper_parameters["mini_batch_size"]
        self.epsilon_init       = hyper_parameters["epsilon_init"]
        self.epsilon_decay      = hyper_parameters["epsilon_decay"]
        self.epsilon_min        = hyper_parameters["epsilon_min"]
        self.learning_rate      = hyper_parameters["learning_rate"]
        self.discount_factor    = hyper_parameters["discount_factor"]

        self.network_sync_rate  = hyper_parameters["network_sync_rate"]
        self.loss_fn = nn.MSELoss() # Mean Squared Error
        self.optimizer = None

    def run(self, is_training: bool = True, render_freq: int = 0):
        """
        Runs the agent
        """
        # Initialize the environment
        env = GridMuckEnvV2(size=11, render_mode="human" if render_freq > 0 else None, max_steps=50)
        obs, info = env.reset()

        # Initialize the model
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to(device)

        # Keep track of rewards
        rewards_per_episode = []
        epsilon_history = []

        if is_training:
            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize target network
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Step counter used for syncing target network
            step_counter = 0

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        for episode in itertools.count():
            # Render every x episodes
            if render_freq > 0 and episode % render_freq == 0:
                env.render_mode = "human"
            else:
                env.render_mode = None

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
                        action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()
 
                # Take action
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate reward
                episode_reward += reward

                # Convert state and reward to tensor
                new_state = self._to_tensor_float(new_state)
                reward = self._to_tensor_float(reward)


                if is_training:
                    # Save experience in replay memory
                    memory.append((state, action, new_state, reward, terminated or truncated))

                    # Increment step counter
                    step_counter += 1

                # Move to new state
                state = new_state
                
                if env.render_mode == "human":
                    time.sleep(0.1)

            # Add reward to list
            rewards_per_episode.append(episode_reward)

            # Update epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            # If enough experience has been collected
            if len(memory)>=self.mini_batch_size:

                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Copy policy network into target network
                if step_counter > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_counter = 0

            if episode % 100 == 0:
                # Print average eposide reward over the last 100 episodes
                print(f"Episode: {episode}, Reward: {round(np.mean(rewards_per_episode[-100:]), 2)}, Epsilon: {round(epsilon, 4)}")

            if not is_training:
                break
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transpose the list of experiences and seperate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values
            target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]
        
        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss for the current whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def _to_tensor_float(self, value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    def _to_tensor_long(self, value):
        return torch.tensor(value, dtype=torch.long, device=device)


if __name__ == "__main__":
    agent = Agent("version_2")
    agent.run(is_training=True, render_freq=1000)