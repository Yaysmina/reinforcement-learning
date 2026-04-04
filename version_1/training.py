import numpy as np
from math import log
from game import Map
from Q_learning import Q_learning_agent
import matplotlib.pyplot as plt

np.set_printoptions(precision=1, suppress=True)

def plot_value_table(agent):
    table = agent.get_value_table()
    size = agent.map.size
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the heatmap
    im = plt.imshow(table, cmap='RdYlGn', vmin=-2, vmax=10)
    
    # Add a color bar
    plt.colorbar(im)
    
    # 1. Remove the axis ticks and labels (the numbers on the side)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 2. Add grid lines
    # We set minor ticks at the halfway points (-0.5, 0.5, 1.5...) to draw lines BETWEEN cells
    ax.set_xticks(np.arange(-.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, size, 1), minor=True)
    
    # Style the grid lines (white, solid, slightly thick)
    # ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Hide the "minor" tick marks themselves so only the grid remains
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title("Agent Value Table (Heatmap)")
    plt.show()




# ----- Setup -----

map = Map(size = 50) # <---------- CHANGE MAP SIZE HERE ----------
agent = Q_learning_agent(
    map,
    alpha = 0.2,
    gamma = 0.98,
    epsilon = 0.9,
    )

# update_freq = episodes / 100
update_list = [25, 50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12500,
               15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
episodes = 100000

min_epsilon = 0.10

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

    if episode in update_list:
        print("Episode", episode, "completed")
        print("Current Epsilon", round(agent.epsilon, 2))
        print("Current value table \n", agent.get_value_table(), "\n")

plot_value_table(agent)

# # ----- Agent Plays -----

# agent.epsilon = 0
# map.logging = True
# map.reset()
# step_count = 0

# print("\n" * 2)
# print("---------- Agent plays ----------")
# print("\n" * 2)
# print("Starting state")
# print(map.grid)
# print()
# print("----------------------")
# print()

# state = map.get_state()
# while not map.is_terminal_state():
#     agent.take_step(state, testing = True)
#     state = map.get_state()
#     step_count += 1

# print("\n" * 2)
# print("Agent finished in", step_count, "steps")
# print("")
# print("Final Value Table")
# plot_value_table(agent)