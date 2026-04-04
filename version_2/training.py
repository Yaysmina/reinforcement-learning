import numpy as np
from math import log
import matplotlib.pyplot as plt

from game import Map
from agent import QLearningAgent

np.set_printoptions(precision=1, suppress=True)

def train():
    grid_size = 5
    env = Map(size=grid_size)

    # State size is position_state * stick_state * agent_hp * zombie_hp
    state_size =    (grid_size**2) *      2      *     3    *     3
    
    agent = QLearningAgent(
        state_size=state_size,
        action_size=5,
        alpha=0.2,
        gamma=0.9,
        epsilon=0.9
    )

    episodes = 100
    update_freq = episodes / 10
    min_epsilon = 0.10
    testing_epsilon = 0.05
    agent.epsilon_decay = -log(min_epsilon / agent.epsilon) / episodes

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

        if episode % update_freq == 0:
            print(f"Episode {episode} completed | Epsilon: {round(agent.epsilon, 2)}")

    print("\n--- Agent Evaluation Play ---")
    agent.epsilon = testing_epsilon
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