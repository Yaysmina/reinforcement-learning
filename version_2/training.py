import numpy as np
from math import log
import matplotlib.pyplot as plt

from game import Map
from agent import QLearningAgent

np.set_printoptions(precision=1, suppress=True)

def train():
    grid_size = 25
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

    episodes = 10000
    update_freq = episodes / 10
    min_epsilon = 0.10
    agent.epsilon_decay = -log(min_epsilon / agent.epsilon) / episodes

    print("Starting Training...")

    for episode in range(1, episodes + 1):
        env.reset()
        state = env.get_state()
        done = False
        step_count = 0
        
        # Training loop bounds to maximum 100 steps
        while not done and step_count < 100:
            action = agent.get_action(state)
            reward = env.take_action(action)
            new_state = env.get_state()
            done = env.is_terminal_state()
            
            agent.update(state, action, reward, new_state, done)
            state = new_state
            step_count += 1
            
        agent.decay_epsilon()

        if episode % update_freq == 0:
            print(f"Episode {episode} completed | Epsilon: {round(agent.epsilon, 2)}")

    print("\n--- Agent Evaluation Play ---")
    
    # Disabled logging to prevent spamming the console during multiple tests
    env.logging = False 
    
    agent.epsilon = 0.05 # 5% chance of random action
    amount_of_tests = 100
    wins = 0
    total_steps = 0
    won_steps = 0

    for _ in range(amount_of_tests):
        env.reset()
        state = env.get_state()
        step_count = 0
        done = False
        reward = 0
        
        # Testing loop bounds to maximum 100 steps
        while not done and step_count < 100:
            action = agent.get_action(state)
            reward = env.take_action(action)
            state = env.get_state()
            done = env.is_terminal_state()
            step_count += 1

        # Winning is having done less than 100 steps or ending with a positive reward
        if step_count < 100 or reward > 0:
            wins += 1
            won_steps += step_count
            
        total_steps += step_count

    # Calculate statistics
    win_percentage = (wins / amount_of_tests) * 100
    avg_total_steps = total_steps / amount_of_tests
    avg_won_steps = won_steps / wins if wins > 0 else 0

    # Display results
    print(f"Played {amount_of_tests} games.")
    print(f"Win Percentage: {win_percentage:.2f}%")
    print(f"Average steps (all games): {avg_total_steps:.2f}")
    print(f"Average steps (won games): {avg_won_steps:.2f}")
    print(f"Optimal amount of steps: 34")

if __name__ == "__main__":
    train()