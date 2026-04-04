import numpy as np
from math import log
import matplotlib.pyplot as plt

from game import Map
from agent import QLearningAgent

np.set_printoptions(precision=1, suppress=True)

def plot_value_table(agent: QLearningAgent, grid_size: int):
    table = agent.get_value_table(grid_size)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = plt.imshow(table, cmap='RdYlGn', vmin=-2, vmax=10)
    plt.colorbar(im)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title("Agent Value Table (Heatmap)")
    plt.show()

def train():
    grid_size = 25
    env = Map(size=grid_size)
    
    agent = QLearningAgent(
        state_size=grid_size * grid_size,
        action_size=4,
        alpha=0.2,
        gamma=0.95,
        epsilon=0.9
    )

    episodes = 10000
    min_epsilon = 0.10
    agent.epsilon_decay = -log(min_epsilon / agent.epsilon) / episodes

    update_list = {25, 50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 
                   6000, 8000, 10000}

    print("Starting Training...")

    for episode in range(1, episodes + 1):
        env.reset()
        state = env.get_state()
        done = False
        
        while not done:
            action = agent.get_action(state)
            reward = env.take_action(action)
            new_state = env.get_state()
            done = env.is_terminal_state()
            
            agent.update(state, action, reward, new_state, done)
            state = new_state
            
        agent.decay_epsilon()

        if episode in update_list:
            print(f"Episode {episode} completed | Epsilon: {round(agent.epsilon, 2)}")

    plot_value_table(agent, grid_size)

    print("\n--- Agent Evaluation Play ---")
    agent.epsilon = 0.0 
    env.logging = True
    env.reset()
    
    state = env.get_state()
    step_count = 0
    done = False

    while not done:
        action = agent.get_action(state)
        env.take_action(action)
        state = env.get_state()
        done = env.is_terminal_state()
        step_count += 1

    print(f"\nAgent finished in {step_count} steps.")

if __name__ == "__main__":
    train()