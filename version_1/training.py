import numpy as np
from math import log
from game import Map
from Q_learning import Q_learning_agent
import matplotlib.pyplot as plt

np.set_printoptions(precision=1, suppress=True)

def plot_value_table(agent):
    table = agent.get_value_table()
    
    plt.figure(figsize=(10, 8))
    
    # cmap='RdYlGn' provides the Red -> Yellow -> Green gradient
    # vmin/vmax ensures 0 is pure red and 1 is pure green
    im = plt.imshow(table, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add a color bar on the side
    plt.colorbar(im)
    
    # Optional: Add the actual numbers inside the squares
    for i in range(agent.map.size):
        for j in range(agent.map.size):
            val = round(table[i, j], 2)
            # Use white text for very red/green blocks and black for yellow for readability
            color = "white" if abs(val - 0.5) > 0.3 else "black"
            plt.text(j, i, val, ha="center", va="center", color=color, fontsize=8)

    plt.title("Agent Value Table (Heatmap)")
    plt.show()

# ----- Setup -----

map = Map(size = 15, walls = 50)
agent = Q_learning_agent(
    map,
    alpha = 0.2,
    gamma = 0.95,
    epsilon = 0.9,
    )
episodes = 1000
update_freq = episodes / 10
min_epsilon = 0.20

agent.epsilon_decay = -log(min_epsilon / agent.epsilon) / episodes

map.logging = False

print("Starting state")
print(map.grid)
print()
print("----------------------")
print()

# ----- Training -----

for episode in range(1, episodes + 1):
    map.reset()
    state = map.get_state()
    done = False
    
    while not done:
        # 1. Choose and take action
        action = agent.get_action(state)
        reward = map.take_action(action)
        new_state = map.get_state()
        done = map.is_terminal_state() # Check if this move ended the game
        
        # 2. Update Q-table (passing 'done' to the function)
        agent.update_q_table(state, action, reward, new_state, done)
        
        state = new_state
        
    agent.update_epsilon()

    if episode % update_freq == 0:
        print("Episode", episode, "completed")
        print("Current Epsilon", round(agent.epsilon, 2))
        print("Current value table \n", agent.get_value_table(), "\n")


# ----- Agent Plays -----

agent.epsilon /= 10
map.logging = True
map.reset()
step_count = 0

print("\n" * 2)
print("---------- Agent plays ----------")
print("\n" * 2)
print("Starting state")
print(map.grid)
print()
print("----------------------")
print()

# state = map.get_state()
# while not map.is_terminal_state():
#     agent.take_step(state, testing = True)
#     state = map.get_state()
#     step_count += 1

# print("\n" * 2)
# print("Agent finished in", step_count, "steps")
# print("")
# print("Final Value Table")
plot_value_table(agent)