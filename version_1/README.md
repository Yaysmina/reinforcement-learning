# GridMuck RL - Version 1

**Version 1: The Baseline Q-Learning Experiment**

This directory contains the first version of **GridMuck**, created as an experimental first step in my project. The goal of this version is to successfully implement a basic Reinforcement Learning (RL) algorithm and prove that it works in a highly simplified environment before introducing more complex mechanics.

## The V1 Environment

To keep the RL logic easy to track and debug, the game logic, state space, action space, and map are deliberately kept as simple as possible:
* **The Map:** A scalable, variable-sized grid (e.g., 25x25). 
* **The Entities:** Just an Agent (starts in the middle) and a Resource (located at the bottom right).
* **Action Space:** 4 basic movement commands (Up, Down, Left, Right).
* **State Space:** A simple integer representing the agent's current coordinate on the grid.

## The RL Approach

This version uses standard tabular **Q-Learning** updated via Temporal Difference learning.
* Because the state space is just the grid coordinates, the Q-table size is simply `(Total Grid Cells) x (4 Actions)`. 
* For example, on a 25x25 map, the agent easily navigates a table of 2,500 state-action pairs (625 states × 4 actions).
* **Results:** The agent trains in just a couple of seconds and successfully finds the absolute shortest path to the resource. (During testing, the agent also easily learned to navigate around random maze-like walls using this exact setup).

## Looking Forward (Context for V2)

This extremely simple V1 acts as a baseline. Proving that the algorithm can quickly map a simple grid is necessary before expanding the state space. 

In **Version 2**, the complexity of the game will increase significantly to test the limits of the agent:
* **New Actions:** Adding an "attack" action, which works both on the resource and the zombie.
* **New Entities:** Adding a non-moving Zombie.
* **Health Stats:** Both the agent and zombie will have a health stat.
* **Minimal Inventory:** The agent can have a stick.
* **Complex State Space:** Because of introducing health stats and the inventory,
the state space won't be just the position of the agent.

## Files in this Version
* `game.py`: The basic Gridworld logic.
* `agent.py`: The purely tabular Q-Learning algorithm.
* `train.py`: The script to train the agent and visualize the learned Q-table heatmap.
* `play.py`: A script to manually play the grid to test coordinates and mechanics.