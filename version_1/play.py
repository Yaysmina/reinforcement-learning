from game import Map, Action

def play_game():
    env = Map(size=5, logging=False)
    env.reset()
    
    command_mapping = {
        "up": Action.UP,
        "z": Action.UP,
        "right": Action.RIGHT,
        "d": Action.RIGHT,
        "down": Action.DOWN,
        "s": Action.DOWN,
        "left": Action.LEFT,
        "q": Action.LEFT
    }

    print("Welcome to Gridworld!")
    print("Commands: 'up' (z), 'right' (d), 'down' (s), 'left' (q), 'quit' (x)")
    
    done = False
    total_reward = 0.0

    while not done:
        print("\nCurrent Map:")
        print(env.grid)
        
        user_input = input("Enter move: ").strip().lower()
        
        if user_input in ['quit', 'x']:
            print("Exiting game.")
            break
            
        if user_input not in command_mapping:
            print("Invalid input. Try again.")
            continue
            
        action = command_mapping[user_input]
        reward = env.take_action(action)
        total_reward += reward
        
        print(f"Moved {Action(action).name}. Reward received: {reward:.1f}")
        
        done = env.is_terminal_state()

    if done:
        print("\nCurrent Map:")
        print(env.grid)
        print(f"\nGame Over! You found the resource. Total Reward: {total_reward:.1f}")

if __name__ == "__main__":
    play_game()