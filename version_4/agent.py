import time
import math
import os
import csv
import sys
import torch
from environment import GridMuckEnvV4
from model import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)

class Agent:
    """
    Agent class
    """
    def __init__(self, seed: int, hyper_parameters_set: str = "version_4"):
        """
        Initializes the agent

        Args:
            seed: Random seed used for the training environment.
            hyper_parameters_set: Name of the hyper parameter set to load.
        """
        with open("hyper_parameters.yml", "r") as file:
            all_hyper_parameters = yaml.safe_load(file)
            hyper_parameters = all_hyper_parameters[hyper_parameters_set]

        # Environment parameters
        self.env_size = hyper_parameters["env_size"]
        self.max_steps = hyper_parameters["max_steps"]

        # Experiment parameters
        self.experiment_name = hyper_parameters["experiment_name"]
        self.seed = seed

        # Hyper parameters
        self.replay_memory_size = hyper_parameters["replay_memory_size"]
        self.mini_batch_size    = hyper_parameters["mini_batch_size"]
        self.epsilon_init       = hyper_parameters["epsilon_init"]
        self.decay_rate         = hyper_parameters["decay_rate"]
        self.epsilon_min        = hyper_parameters["epsilon_min"]
        self.min_lr             = hyper_parameters["min_lr"]
        self.initial_lr         = hyper_parameters["initial_lr"]
        self.decay_episodes     = hyper_parameters["decay_episodes"]
        self.discount_factor    = hyper_parameters["discount_factor"]

        self.network_sync_rate  = hyper_parameters["network_sync_rate"]
        self.loss_fn = nn.MSELoss() # Mean Squared Error
        self.optimizer = None

        # Evaluation hyper parameters
        self.eval_seed_start   = hyper_parameters["eval_seed_start"]
        self.eval_episode_count = hyper_parameters["eval_episode_count"]
        self.eval_freq         = hyper_parameters["eval_freq"]
        self.human_rendering   = hyper_parameters["human_rendering"]

        # Keep track of best model (lower is better)
        self.best_performance = float("inf")

    def run(self):
        """
        Runs the agent.

        Two separate environments are instantiated:
          - train_env: used exclusively for stochastic training and replay buffer
            data collection.
          - eval_env:  kept idle, stepped only during evaluation checkpoints to
            prevent polluting the training environment's state or RNG.
        """
        # Dual environment architecture
        train_env = GridMuckEnvV4(size=self.env_size, render_mode=None, max_steps=self.max_steps)
        eval_env = GridMuckEnvV4(size=self.env_size, render_mode=None, max_steps=self.max_steps)
        train_env.reset(seed=self.seed)

        # Initialize the model
        num_states = train_env.observation_space.shape[0]
        num_actions = train_env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to(device)
        print("Model is on device:", next(policy_dqn.parameters()).device)

        # Initialize replay memory
        memory = ReplayMemory(self.replay_memory_size)

        # Initialize target network
        target_dqn = DQN(num_states, num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Step counter used for syncing target network
        step_counter = 0

        # Policy network optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.initial_lr)

        # Initialize the experiment-specific CSV log file
        self._init_csv_log()

        epsilon = self.epsilon_init

        for episode in itertools.count():
            # Initialize the environment
            state, _ = train_env.reset()
            state = self._to_tensor_float(state)
            terminated = False
            truncated = False

            # Play the game once
            while not (terminated or truncated):
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = train_env.action_space.sample()
                    action = self._to_tensor_long(action)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()

                # Take action
                new_state, reward, terminated, truncated, info = train_env.step(action.item())

                # Convert state and reward to tensor
                new_state = self._to_tensor_float(new_state)
                reward = self._to_tensor_float(reward)

                # Save experience in replay memory
                memory.append((state, action, new_state, reward, terminated or truncated))

                # Increment step counter
                step_counter += 1

                # Move to new state
                state = new_state

            # Update epsilon using the decay formula
            epsilon = max(self.epsilon_min, 1.0 / math.sqrt(1.0 + self.decay_rate * episode))

            # Update learning rate using linear decay
            lr = max(self.min_lr, self.initial_lr - (episode / self.decay_episodes) * (self.initial_lr - self.min_lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # If enough experience has been collected
            if len(memory) >= self.mini_batch_size:
                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Copy policy network into target network
                if step_counter > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_counter = 0

            # Run the fixed-seed benchmark sweep at each evaluation milestone
            if episode > 0 and (episode < self.eval_freq and episode % (self.eval_freq // 10) == 0) or (episode % self.eval_freq == 0):
                win_rate, mean_episode_length = self.evaluate(policy_dqn, eval_env)
                self._log_benchmark_row(episode, win_rate, mean_episode_length, epsilon)

                # Keep track of best model (lower is better)
                # Scored by derivation from optimal win rate and optimal mean episode length
                score = 100 - (100*win_rate) + mean_episode_length - 12
                if score < self.best_performance:
                    self.best_performance = score
                    print(f"New best performance! Score: {self.best_performance:.1f}. Saving model...")

                    # Create the directory if it doesn't exist
                    checkpoint_dir = os.path.join("checkpoints", self.experiment_name)
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # Define the save path
                    save_path = os.path.join(checkpoint_dir, f"best_model_seed_{self.seed}.pt")
                    
                    # Save the model's state dictionary
                    torch.save(policy_dqn.state_dict(), save_path)
                    print(f"Model saved to {save_path}")




    def evaluate(self, policy_dqn, eval_env):
        """
        Runs a dedicated 100-game benchmark sweep on the eval environment using a
        fixed set of seeds. The model is put in evaluation mode with gradients
        disabled and actions are chosen purely greedily (no epsilon exploration).

        Returns: (win_rate, mean_episode_length)
        """
        # Put the model in evaluation mode and disable gradients
        policy_dqn.eval()

        if self.human_rendering:
            # Pick a random test run to display in human render mode
            render_seed = random.randint(self.eval_seed_start, self.eval_seed_start + self.eval_episode_count - 1)

        wins = 0
        episode_lengths = []

        with torch.no_grad():
            for seed in range(self.eval_seed_start, self.eval_seed_start + self.eval_episode_count):
                if self.human_rendering:
                    # Render this run in human mode so the agent's behavior is visible
                    if seed == render_seed:
                        eval_env.render_mode = "human"
                    else:
                        eval_env.render_mode = None

                state, _ = eval_env.reset(seed=seed)
                state = self._to_tensor_float(state)
                terminated = False
                truncated = False
                steps = 0

                while not (terminated or truncated):
                    # Purely greedy action selection
                    action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()

                    new_state, reward, terminated, truncated, info = eval_env.step(action.item())
                    state = self._to_tensor_float(new_state)
                    steps += 1

                    if self.human_rendering and eval_env.render_mode == "human":
                        time.sleep(0.2)

                episode_lengths.append(steps)
                if eval_env.zombie_hp <= 0:
                    wins += 1

        # Restore the model to training mode
        policy_dqn.train()

        win_rate = wins / self.eval_episode_count
        mean_episode_length = float(np.mean(episode_lengths))
        return win_rate, mean_episode_length

    def _init_csv_log(self):
        """
        Ensures the target log directory exists and writes the CSV header at the
        start of the run. Logs are written to:
            logs/<experiment_name>/run_seed_<seed>.csv
        """
        log_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, f"run_seed_{self.seed}.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["training_episode", "win_rate", "mean_episode_length", "epsilon"])

    def _log_benchmark_row(self, training_episode: int, win_rate: float,
                           mean_episode_length: float, epsilon: float):
        """
        Appends a single row to the experiment CSV at each evaluation milestone
        and prints it to the console.
        """
        if not hasattr(self, "csv_path"):
            self._init_csv_log()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([training_episode, round(win_rate, 4),
                             round(mean_episode_length, 2), round(epsilon, 4)])

        print(f"Episode: {training_episode}, Win rate: {round(win_rate, 4)}, "
              f"Mean episode length: {round(mean_episode_length, 2)}, Epsilon: {round(epsilon, 4)}")

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
    if len(sys.argv) != 2:
        print("Usage: python3 agent.py <seed>")
        sys.exit(1)

    seed = int(sys.argv[1])
    agent = Agent(seed, "version_4")
    agent.run()