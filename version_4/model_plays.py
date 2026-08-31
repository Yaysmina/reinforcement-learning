import time
import pygame
import torch
from environment import GridMuckEnvV4
from model import DQN
from visualization import Visualization

# ================= Configuration =================
MODEL_PATH = "checkpoints/6-decreasing-lr/best_model_seed_1.pt"  # Update to your best model path
ENV_SIZE = 9
MAX_STEPS = 50
STEP_DELAY = 0.5  # Seconds between moves
# =================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Initialize environment & visualization
    env = GridMuckEnvV4(size=ENV_SIZE, max_steps=MAX_STEPS)
    vis = Visualization(cell_size=60)
    vis.show()

    # 2. Load trained DQN model
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model = DQN(num_states, num_actions).to(device)

    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("\n🎮 Playback started! Close the window or press ESC in the terminal to exit.\n")

    episode_num = 1

    try:
        while vis.running:
            state, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            # Render initial state
            vis.render(env)
            time.sleep(1)

            # Game loop for one episode
            while not (terminated or truncated) and vis.running:
                # Greedy action from policy network
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = model(state_tensor).squeeze(0).argmax().item()

                # Step the environment
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Render current step
                vis.render(env)
                time.sleep(STEP_DELAY)

            if not vis.running:
                break

            # Determine outcome message for the overlay
            if env.zombie_hp <= 0:
                status_msg = f"EPISODE {episode_num}: VICTORY!"
            elif env.agent_hp <= 0:
                status_msg = f"EPISODE {episode_num}: DEFEATED!"
            else:
                status_msg = f"EPISODE {episode_num}: TIMED OUT"

            print(f"{status_msg} | Total Reward: {total_reward:.2f} | Steps: {env.current_step}")

            # Show the overlay banner for 1.5 seconds before starting the next game
            start_pause = time.time()
            while time.time() - start_pause < 1.5 and vis.running:
                vis.render(env, status_message=status_msg)

            episode_num += 1

    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        vis.close()


if __name__ == "__main__":
    main()