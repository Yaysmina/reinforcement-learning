import time
from environment import GridMuckEnvV2
from visualization import Visualization

def play_game():
    env = GridMuckEnvV2(size=9, max_steps=1000, logging=False)
    obs, info = env.reset()
    vis = Visualization(cell_size=80)
    vis.show()

    total_reward = 0.0
    done = False
    final_msg = ""

    # --- MAIN GAME LOOP ---
    while not done and vis.running:
        vis.render(env)
        
        action = vis.get_human_action()
        if action is None: break # Window closed or X pressed

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # --- DETERMINE OUTCOME ---
    if done:
        if env.agent_hp <= 0:
            final_msg = f"YOU DIED! Score: {total_reward:.1f}"
        elif env.zombie_hp <= 0:
            final_msg = f"ZOMBIE SLAIN! Score: {total_reward:.1f}"
        elif env.current_step >= env.max_steps:
            final_msg = f"OUT OF TIME! Score: {total_reward:.1f}"
        
        print(f"Game Over. {final_msg}")

        # --- FINAL STATIC LOOP (Wait for user to close window) ---
        while vis.running:
            vis.render(env, status_message=final_msg)
            # We don't call get_human_action here because we don't want to process moves
            # The vis.render method already handles the 'QUIT' event internally
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_x):
                    vis.running = False

    vis.close()

if __name__ == "__main__":
    play_game()