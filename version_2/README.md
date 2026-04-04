# GridMuck RL - Version 2

**Version 2: Expanding the State Space**

This directory contains the second version of **GridMuck**, in which I made the game logic and state space more complexed. The goal of this version is to see if the same reinforcement learning algorithm from the previous version is able to deal with a complexer state space. The result is a definite YES!

## The V2 Game
In the second version of GridMuck, the goal of the game is now to defeat a zombie.
Both the agent and the zombie start with 2 HP.
When the agent moves next to the zombie, the zombie will deal when Damage.
When standing next to a zombie and doing the "ATTACK" action, the agent will deal 1 Damage.
The point is that the agent cannot defeat the zombie like this, because the zombie attacks after the agent ends up next to it, dealing the first punch.
This results in the agent dying before it can kill the zombie, losing the game.
To beat the zombie, the agent must first walk up to the tree and use the "ATTACK" action to get a stick.
With this stick the agent now deals 2 Damage to the zombie and is thus able to beat it, winning the game.

# Features
* **The Map:** A scalable, variable-sized grid (e.g., 25x25). 
* **The Entities:** An agent, a tree and a stationary zombie.
* **Action Space:** The 4 movement actions and now also an "ATTACK" actions, which works both on the tree and the zombie.
* **State Space:** The agent's position, agent's hp (0-2), zombie's hp (0-2), and if the agent has a stick or not.
* **State Space Size:** The grid size * 3 * 3 * 2 -> Grid size * 18

## The RL Approach

The same as before, the agent was able to successfully complete the game as fast as possible even with a 15x15 grid. It found that it needs to go to the tree, attack it, go to the zombie, attack it.

## Looking Forward (Context for V3)

In the next version of this project, I will significantly increase the complexity of the game.

Because of the increased complexity, I will probably need to both optimize and improve the Q-Learning algorithm.

In **Version 3**, the game will be an actual playable game, with a different space state and different reward structure:
* **Harder Enemy:** The zombie will now be able to move and chase the agent.
* **Harder Crafting:** The agent will need to collect 5 sticks to make a wooden sword.
* **Different Rewards:** The agent will be rewarded in a totally different way, namely staying alive as long as possible.
* **Improved Visualization:** The game will be displayed using pygame, both for manual playtesting and debugging the agent.
* **Improved Performance:** I will try to make both the game, and training algorithm more performant. Which means that during training pygame will not be used, except for visual displays of how the agent plays for each training update. This will be very useful for debugging and visually checking agent performance.

## Files in this Version
* `game.py`: The game logic.
* `agent.py`: The purely tabular Q-Learning algorithm.
* `train.py`: The script to train the agent and see testing performance stats.
* `play.py`: A script to manually play the game yourself.