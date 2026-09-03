# GridMuck — Deep Q-Learning Agent (Version 4)

A reinforcement learning project that trains a **Deep Q-Network (DQN)** agent to play *GridMuck*, a small grid-based survival game. The agent must navigate a 2D grid, collect a stick from a tree, and defeat a zombie while avoiding damage.

This repository documents the **final version (v4)** of the project, which is the complete and definitive implementation. Earlier iterations (`version_1`, `version_2`, `version_3`) are kept for historical reference only and are superseded by `version_4`.

---

## Table of Contents

- [Overview](#overview)
- [The Game](#the-game)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train an Agent](#train-an-agent)
  - [Watch a Trained Model Play](#watch-a-trained-model-play)
  - [Play Manually](#play-manually)
  - [Plot Results](#plot-results)
- [Configuration](#configuration)
- [Reinforcement Learning Design](#reinforcement-learning-design)
- [Experiments & Results](#experiments--results)
- [License](#license)

---

## Overview

Version 4 implements a complete DQN training pipeline built on the **Gymnasium** API:

- A custom [`GridMuckEnvV4`](version_4/environment.py) environment with a 9-dimensional normalized observation space and 5 discrete actions.
- A feed-forward [`DQN`](version_4/model.py) neural network (two hidden layers of 256 units).
- A [`ReplayMemory`](version_4/experience_replay.py) experience-replay buffer.
- A [`Agent`](version_4/agent.py) that trains the policy network, maintains a target network, and periodically evaluates performance on a fixed-seed benchmark.
- A [`Visualization`](version_4/visualization.py) Pygame renderer for human play and model playback.
- Experiment logging to CSV and a [`plot_results.py`](version_4/plot_results.py) script to compare multiple hyper-parameter variants.

---

## The Game

*GridMuck* is a turn-based grid game. The agent starts in the center of a `5×5` (configurable) grid. A **tree** and a **zombie** are placed in two random, opposite outer quadrants.

**Goal:** defeat the zombie before the agent dies or runs out of steps.

### Entities

| Entity | Description |
|--------|-------------|
| Agent  | Controlled by the human or the DQN policy. Starts with 2 HP. |
| Tree   | Standing next to it and attacking grants the agent a **stick**. |
| Zombie | Moves toward the agent with 50% probability each step and attacks when adjacent. Starts with 2 HP. |

### Actions (5 discrete)

| Action | Effect |
|--------|--------|
| `UP` / `DOWN` / `LEFT` / `RIGHT` | Move one cell (blocked by non-empty cells). |
| `ATTACK` | If next to the tree, pick up a stick; if next to the zombie, deal damage (2 with a stick, otherwise 1). |

### Observation Space (9 values, normalized)

| Index | Meaning |
|-------|---------|
| 0, 1  | Relative X / Y position of the agent |
| 2, 3  | Relative X / Y distance to the zombie |
| 4, 5  | Relative X / Y distance to the tree |
| 6     | Normalized agent HP |
| 7     | Normalized zombie HP |
| 8     | Binary flag: does the agent have a stick? |

### Reward Function

| Event | Reward |
|-------|--------|
| Time step penalty | `-0.3` |
| Picking up the stick | `+5.0` (once) |
| Agent dies | `-10.0` |
| Zombie dies | `+10.0` |

---

## Project Structure

```
version_4/
├── agent.py              # DQN training loop, evaluation, CSV logging
├── environment.py        # GridMuckEnvV4 Gymnasium environment
├── model.py              # DQN neural network
├── experience_replay.py  # ReplayMemory buffer
├── visualization.py      # Pygame renderer
├── play.py               # Manual human-play script
├── model_plays.py        # Watch a trained model play
├── plot_results.py       # Compare experiment results
├── hyper_parameters.yml  # All hyper-parameters
├── requirements.txt      # Python dependencies
├── checkpoints/          # Saved best models per experiment
│   └── <experiment>/best_model_seed_<seed>.pt
└── logs/                 # CSV logs per experiment
    ├── <experiment>/run_seed_<seed>.csv
    └── results.png       # Generated comparison plot
```

---

## Installation

Requires **Python 3.9+**. Install the dependencies from within the `version_4` directory:

```bash
cd version_4
pip install -r requirements.txt
```

Dependencies:

- `numpy>=1.24.0`
- `pygame>=2.5.0`
- `gymnasium>=1.0.0`
- `torch>=2.0.0`
- `PyYAML>=6.0`
- `matplotlib>=3.7.0`
- `pandas>=2.0.0`

> **Note:** All scripts must be run from inside the `version_4` directory, since they load `hyper_parameters.yml` and write to `logs/` and `checkpoints/` using relative paths.

---

## Usage

### Train an Agent

Train a DQN agent with a given random seed:

```bash
cd version_4
python3 agent.py <seed>
```

For example:

```bash
python3 agent.py 1
```

During training the agent:

1. Collects experience in a replay buffer using ε-greedy exploration.
2. Optimizes the policy network on random mini-batches.
3. Periodically copies the policy network into the target network.
4. Runs a fixed-seed, 100-episode evaluation sweep at each milestone.
5. Saves the best model (by a combined win-rate / episode-length score) to `checkpoints/<experiment>/best_model_seed_<seed>.pt`.
6. Logs each evaluation to `logs/<experiment>/run_seed_<seed>.csv`.

### Watch a Trained Model Play

Playback a saved checkpoint with greedy action selection:

```bash
cd version_4
python3 model_plays.py
```

Edit the `MODEL_PATH` constant at the top of [`model_plays.py`](version_4/model_plays.py) to point at the checkpoint you want to watch. Close the window or press `ESC` to stop.

### Play Manually

Play the game yourself with the keyboard:

```bash
cd version_4
python3 play.py
```

**Controls:** arrow keys or `Z`/`Q`/`S`/`D` to move, `SPACE` or `A` to attack, `X`/`ESC` to quit.

### Plot Results

Compare the average performance of all experiment variants found in `logs/`:

```bash
cd version_4
python3 plot_results.py [max_episodes] [filter]
```

- `max_episodes` — optional X-axis cap in `{number}k` form (e.g. `30k`).
- `filter` — optional suffix filter; prefix with `no_` to exclude (e.g. `no_bad` drops variants ending in `bad`).

The script averages the seeded runs of each variant and produces a two-panel figure (win rate and mean episode length) saved to `logs/results.png`.

---

## Configuration

All hyper-parameters live in [`hyper_parameters.yml`](version_4/hyper_parameters.yml) under the `version_4` key:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env_id` | `GridMuck-v4` | Environment identifier |
| `env_size` | `5` | Grid size (N×N) |
| `max_steps` | `50` | Max steps per episode |
| `experiment_name` | `test` | Sub-folder name for logs/checkpoints |
| `replay_memory_size` | `10000` | Replay buffer capacity |
| `mini_batch_size` | `32` | Mini-batch size for training |
| `epsilon_init` | `1.0` | Initial exploration rate |
| `decay_rate` | `0.01` | ε decay rate (`ε = 1/√(1 + decay·episode)`) |
| `epsilon_min` | `0.02` | Minimum exploration rate |
| `network_sync_rate` | `100` | Steps between target-network syncs |
| `initial_lr` | `0.001` | Initial learning rate |
| `min_lr` | `0.0001` | Minimum learning rate |
| `decay_episodes` | `50000` | Episodes over which LR linearly decays |
| `discount_factor` | `0.98` | Discount factor γ |
| `eval_seed_start` | `1000` | First seed of the evaluation sweep |
| `eval_episode_count` | `100` | Number of evaluation episodes |
| `eval_freq` | `1000` | Episodes between evaluations |
| `human_rendering` | `True` | Render one eval run in human mode |

---

## Reinforcement Learning Design

The agent uses **Deep Q-Learning** with two key stabilisation techniques:

1. **Experience Replay** — transitions `(state, action, new_state, reward, done)` are stored in a fixed-size buffer and sampled uniformly in mini-batches, decorrelating training samples.
2. **Target Network** — a separate network, synced every `network_sync_rate` steps, provides stable Q-value targets.

**Exploration** uses ε-greedy with a decaying ε:

```
ε = max(ε_min, 1 / √(1 + decay_rate · episode))
```

**Learning rate** is linearly annealed from `initial_lr` to `min_lr` over `decay_episodes`.

**Evaluation** runs a dedicated 100-episode sweep on a separate environment using fixed seeds and purely greedy action selection. The best model is selected by the score:

```
score = 100 − (100 · win_rate) + mean_episode_length − 12
```

(lower is better), balancing a high win rate against a short episode length.

---

## Experiments & Results

The `logs/` directory contains the results of several hyper-parameter experiments, each run across multiple seeds:

| Experiment | Description |
|------------|-------------|
| `1-baseline` | Default configuration |
| `2-faster-epsilon-decay` | Faster ε decay |
| `3-bigger-replay-memory_bad` | Larger replay buffer (underperformed) |
| `4-slower-sync-rate` | Slower target-network sync |
| `5-higher-step-penalty` | Stronger time-step penalty |
| `6-decreasing-lr` | Decreasing learning rate (best model used for playback) |
| `7-linear-epsilon-decay_bad` | Linear ε decay (underperformed) |

Each experiment folder contains `run_seed_<seed>.csv` files with columns `training_episode, win_rate, mean_episode_length, epsilon`. The best-performing checkpoints are stored under `checkpoints/`, and the aggregated comparison plot is generated as `logs/results.png`.

---

## License

This project is provided for educational and research purposes.