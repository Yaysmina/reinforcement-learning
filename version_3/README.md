# GridMuck RL - Version 3

**Version 3: From Tables to Neural Networks — Deep Q-Learning**

This directory contains the third version of **GridMuck**, marking a major architectural shift from tabular Q-Learning to **Deep Q-Learning (DQN)**. Instead of storing Q-values in a lookup table, a neural network approximates the Q-function, enabling the agent to handle continuous state spaces and scale to more complex environments.

## The V3 Game

The core game remains similar to Version 2: the agent must collect a stick from a tree, then use it to defeat a zombie. This simple loop tests the agent's ability to learn multi-step planning (explore → collect tool → fight).

### Entity Layout

| Entity  | Starting Position | Description |
|---------|-------------------|-------------|
| Agent   | Center of grid    | Controlled by the RL agent (or human) |
| Tree    | Bottom-right corner | Grants a stick when attacked while adjacent |
| Zombie  | Top-middle of grid | Deals 1 HP damage to the agent when adjacent; requires the stick to defeat |

### Action Space (5 Discrete Actions)

| Action | Keybind     | Effect |
|--------|-------------|--------|
| `UP`   | ↑ / Z       | Move the agent up one cell |
| `DOWN` | ↓ / S       | Move the agent down one cell |
| `LEFT` | ← / Q       | Move the agent left one cell |
| `RIGHT`| → / D       | Move the agent right one cell |
| `ATTACK`| Space / A  | If next to a tree → gain a stick. If next to a zombie → deal damage (1 without stick, 2 with stick) |

### Observation Space (5-D Normalized Vector)

Instead of a discrete integer representing grid position, the agent receives a continuous 5-element vector:

| Index | Feature    | Range  | Description |
|-------|------------|--------|-------------|
| 0     | `agent_x`  | [0, 1] | Agent's X coordinate, normalized by grid width |
| 1     | `agent_y`  | [0, 1] | Agent's Y coordinate, normalized by grid height |
| 2     | `has_stick`| {0, 1}  | Binary flag (0 or 1) indicating stick ownership |
| 3     | `agent_hp` | [0, 1] | Agent's HP, normalized by max HP (2) |
| 4     | `zombie_hp`| [0, 1] | Zombie's HP, normalized by max HP (2) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Time step penalty | -0.1 (per step) |
| Collecting a stick | +5.0 (one-time bonus) |
| Defeating the zombie | +10.0 |
| Agent dies | -10.0 |

The agent is incentivized to act efficiently (negative time penalty) while learning the correct sequence: get the stick → attack the zombie.

## The RL Approach: Deep Q-Learning

This version replaces the tabular Q-table with a **Deep Q-Network (DQN)**, a neural network that approximates the optimal Q-function.

### Neural Network Architecture ([`model.py`](version_3/model.py))

```
Input (5) → FC(256) → ReLU → FC(256) → ReLU → FC(5) → Q-values
```

- **Input layer:** 5 neurons (matching the observation space)
- **Hidden layers:** Two fully-connected layers with 256 neurons each and ReLU activation
- **Output layer:** 5 neurons (one Q-value per action)

### Key DQN Components

1. **Experience Replay** ([`experience_replay.py`](version_3/experience_replay.py))  
   A replay buffer stores past transitions `(state, action, next_state, reward, done)`. During training, random mini-batches are sampled from this buffer, breaking temporal correlations and stabilizing learning.

2. **Target Network**  
   A separate `target_dqn` network maintains stable Q-targets. Its weights are periodically copied from the policy network every `network_sync_rate` steps, reducing harmful feedback loops.

3. **Epsilon-Greedy Exploration**  
   The agent starts with a high exploration rate (`epsilon = 1.0`) that decays exponentially (`decay = 0.99995`) over time, transitioning from random exploration to greedy exploitation.

### Hyperparameters ([`hyper_parameters.yml`](version_3/hyper_parameters.yml))

| Parameter | Value | Description |
|-----------|-------|-------------|
| `replay_memory_size` | 10,000 | Maximum number of stored experiences |
| `mini_batch_size` | 32 | Number of experiences sampled per training step |
| `epsilon_init` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.99995 | Exponential decay rate per episode |
| `epsilon_min` | 0.02 | Minimum exploration rate |
| `network_sync_rate` | 10 | Steps between target network syncs |
| `learning_rate` | 0.0001 | Adam optimizer learning rate |
| `discount_factor` | 0.98 | Future reward discount (γ) |

### Why DQN over Tabular Q-Learning?

- The observation space is now a **continuous 5D vector** — a tabular approach would require discretization, losing precision or exploding the state space.
- The neural network **generalizes** across similar states, enabling the agent to behave sensibly even in states it hasn't explicitly visited.
- DQN with replay memory and a target network provides **more stable and sample-efficient** learning compared to standard Q-Learning updates.

## Files in this Version

| File | Purpose |
|------|---------|
| [`environment.py`](version_3/environment.py) | Gymnasium-compatible environment (`GridMuckEnvV2`) with full game logic |
| [`agent.py`](version_3/agent.py) | DQN agent with training loop, optimization, and evaluation |
| [`model.py`](version_3/model.py) | PyTorch neural network definition (DQN) |
| [`experience_replay.py`](version_3/experience_replay.py) | Replay memory buffer for experience replay |
| [`visualization.py`](version_3/visualization.py) | Pygame-based visualizer with grid display and info panel |
| [`play.py`](version_3/play.py) | Script for human playtesting using keyboard controls |
| [`hyper_parameters.yml`](version_3/hyper_parameters.yml) | Centralized configuration for all training hyperparameters |
| [`requirements.txt`](version_3/requirements.txt) | Python dependencies |

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)

### Installation

1. **Navigate to the project directory**
   ```bash
   cd version_3
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   - **Windows (CMD):** `.venv\Scripts\activate.bat`
   - **Linux / macOS:** `source .venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This installs:
   - [`PyTorch`](https://pytorch.org/) — Deep learning framework for the neural network
   - [`Gymnasium`](https://gymnasium.farama.org/) — Standardized RL environment API
   - [`NumPy`](https://numpy.org/) — Numerical computing
   - [`Matplotlib`](https://matplotlib.org/) — Plotting (for potential logging visualizations)
   - [`Pygame`](https://www.pygame.org/) — Game visualization and rendering
   - [`PyYAML`](https://pyyaml.org/) — YAML configuration file parsing

### Running the Agent

#### Train the DQN Agent

Run [`agent.py`](version_3/agent.py) directly to start training:

```bash
python agent.py
```

This launches training with the default hyperparameters. The agent will print average episode rewards every 100 episodes and render a Pygame visualization every 1,000 episodes so you can visually inspect its progress.

#### Play the Game Manually

```bash
python play.py
```

Use the keyboard controls to move and attack:

| Key | Action | Alternative Key |
|-----|--------|-----------------|
| ↑ | Move Up | Z |
| ↓ | Move Down | S |
| ← | Move Left | Q |
| → | Move Right | D |
| Space | Attack | A |
| X / Esc | Quit | — |

## Key Differences from Version 2

| Aspect | Version 2 (Tabular) | Version 3 (DQN) |
|--------|--------------------|-----------------|
| **Algorithm** | Tabular Q-Learning (ε-greedy) | Deep Q-Network with experience replay |
| **State representation** | Discrete integer (grid cell index) | Continuous 5D normalized vector |
| **Q-value storage** | `np.array` table | Neural network (PyTorch) |
| **Training stability** | Simple TD update | Target network + replay buffer |
| **Environment API** | Custom | Gymnasium (`gym.Env`) |
| **Visualization** | Console-based / Matplotlib heatmap | Pygame grid with real-time info panel |
| **Configuration** | Hardcoded in code | External YAML file |
| **Playability** | Limited console input | Full keyboard-controlled Pygame window |
