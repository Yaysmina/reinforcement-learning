import time
from game import Map, Action
from visualization import Visualization


def play_game():
    env = Map(size=9, logging=False)
    env.reset()

    # Create the visualization (disabled by default)
    vis = Visualization(cell_size=80)

    # Terminal command mapping (used when visualization is off)
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
        "a": Action.ATTACK,
    }

    print("Welcome to Gridworld!")
    print("Commands: 'up' (z), 'right' (d), 'down' (s), 'left' (q), 'attack' (a), 'exit' (x)")
    print("Press 'v' during gameplay to toggle Pygame visualization on/off")

    done = False
    total_reward = 0.0

    while not done:
        # --- Render visualization if enabled ---
        if vis.is_visible():
            vis.render(env)

        # --- Get input (Pygame if visible, terminal otherwise) ---
        action = None

        if vis.is_visible():
            # Input through Pygame
            action = vis.get_action(env)
            if action is None and not vis.running:
                # User closed the window or pressed exit
                print("Exiting game.")
                break
            elif action is None:
                # Toggle happened, no action taken this frame
                continue
        else:
            # Input through terminal
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

            if user_input == 'v':
                vis.toggle()
                continue

            if user_input not in command_mapping:
                print("Invalid input. Try again.")
                continue

            action = command_mapping[user_input].value

        old_agent_HP = env.agent_hp
        old_zombie_HP = env.zombie_hp

        reward = env.take_action(action)
        total_reward += reward

        new_agent_HP = env.agent_hp
        new_zombie_HP = env.zombie_hp

        # --- Print action feedback (only in terminal mode) ---
        if not vis.is_visible():
            print("=" * 50)
            print("")
            print(f"Took action {Action(action).name}. Reward received: {reward:.1f}")

            if old_zombie_HP != new_zombie_HP:
                print(f"You attacked and did {old_zombie_HP - new_zombie_HP} damage!")
            if old_agent_HP != new_agent_HP:
                print(f"The zombie attacked and did {old_agent_HP - new_agent_HP} damage!")

            if env.agent_hp <= 0:
                print("You died!")
            elif env.zombie_hp <= 0:
                print("You killed the zombie!")

        done = env.is_terminal_state()

    # --- Game over ---
    if vis.is_visible():
        # Show final state in visualization for 2 seconds
        vis.render(env)
        time.sleep(2)

    if done:
        print("\nFinal State:")
        print("Stick: ", "yes" if env.has_stick else "no")
        print("Agent Health: ", env.agent_hp)
        print("Zombie Health: ", env.zombie_hp)
        print("Current Map:")
        print(env.grid)
        if env.agent_hp <= 0:
            print(f"\nGame Over! You died. Total Reward: {total_reward:.1f}")
        elif env.zombie_hp <= 0:
            print(f"\nGame Over! You killed the zombie. Total Reward: {total_reward:.1f}")

    vis.close()


if __name__ == "__main__":
    play_game()
