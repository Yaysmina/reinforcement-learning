This is the starting version of the game and agent.

The game is a simple grid world with a resource and an agent.
The agent has to find the resource.
- Agent starts in the middle
- Resource is in the bottom right
The map size is variable, and the agent is easily trained even on bigger maps.
For example in a couple seconds the agent found the resource on a 25x25 map,
which results in a Q table of size 2500 (625 x 4).

I tired adding random walls to have a maze-like map,
and the agent could still easily find the shortest path to the resource.

The approach is to use a Q-Learning algorithm to train the agent.
Specifically, there is a Q-table for each state-action pair.
The Q-table is updated using temporal difference learning.