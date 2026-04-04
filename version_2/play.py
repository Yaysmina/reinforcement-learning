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
        "q": Action.LEFT,
        "attack": Action.ATTACK,
        "a": Action.ATTACK
    }

    print("Welcome to Gridworld!")
    print("Commands: 'up' (z), 'right' (d), 'down' (s), 'left' (q), 'attack' (a), 'exit' (x)")
    
    done = False
    total_reward = 0.0

    while not done:
        print("")
        print("Stick: ", "yes" if env.has_stick else "no")
        print("Agent Health: ", env.agent_hp)
        print("Zombie Health: ", env.zombie_hp)
        print("Current Map:")
        print(env.grid)
        
        user_input = input("Enter action: ").strip().lower()
        
        if user_input in ['exit', 'x']:
            print("Exiting game.")
            break
            
        if user_input not in command_mapping:
            print("Invalid input. Try again.")
            continue

        old_agent_HP = env.agent_hp
        old_zombie_HP = env.zombie_hp
            
        action = command_mapping[user_input]
        reward = env.take_action(action)
        total_reward += reward

        new_agent_HP = env.agent_hp
        new_zombie_HP = env.zombie_hp
        
        print("="*50)
        print("")
        print(f"Took action {Action(action).name}. Reward received: {reward:.1f}")

        # Display damage
        if old_zombie_HP != new_zombie_HP:
            print(f"You attacked and did {old_zombie_HP - new_zombie_HP} damage!")
        if old_agent_HP != new_agent_HP:
            print(f"The zombie attacked and did {old_agent_HP - new_agent_HP} damage!")
        
        # Display deaths
        if env.agent_hp <= 0:
            print("You died!")
        elif env.zombie_hp <= 0:
            print("You killed the zombie!")
        
        done = env.is_terminal_state()

    if done:
        print("\nCurrent Map:")
        print("Stick: ", "yes" if env.has_stick else "no")
        print("Agent Health: ", env.agent_hp)
        print("Zombie Health: ", env.zombie_hp)
        print("Current Map:")
        print(env.grid)
        if env.agent_hp <= 0:
            print(f"\nGame Over! You died. Total Reward: {total_reward:.1f}")
        elif env.zombie_hp <= 0:
            print(f"\nGame Over! You killed the zombie. Total Reward: {total_reward:.1f}")

if __name__ == "__main__":
    play_game()