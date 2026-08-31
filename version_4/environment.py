import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum

class Entity(IntEnum):
    EMPTY = 0
    TREE = 1
    AGENT = 2
    ZOMBIE = 3

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    ATTACK = 4

class GridMuckEnvV4(gym.Env):
    """
    A single-class Gymnasium environment for GridMuck (Version 4).
    Ready for Deep Q-Learning.
    """
    
    # Gymnasium metadata
    metadata = {"render_modes": ["human"]}

    # The 4 cardinal directions
    NEIGHBORS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    
    ACTIONS_TO_MOVE = {
        Action.UP: NEIGHBORS[0],
        Action.RIGHT: NEIGHBORS[1],
        Action.DOWN: NEIGHBORS[2],
        Action.LEFT: NEIGHBORS[3]
    }

    def __init__(self, size: int = 7, max_steps: int = 100, logging: bool = False, render_mode: str = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.window = None
        
        self.size = size
        self.max_steps = max_steps
        self.logging = logging

        self.min_x = 0
        self.max_x = size - 1
        self.min_y = 0
        self.max_y = size - 1

        # --- ACTION SPACE ---
        # 5 Discrete actions: Up, Right, Down, Left, Attack
        self.action_space = spaces.Discrete(5)

        # --- OBSERVATION SPACE ---
        # 1D Vector of 9 normalized values:
            # 0: Relative X position (-1.0 to 1.0)
            # 1: Relative Y position (-1.0 to 1.0)
            # 2: Relative X distance to Zombie (-1.0 to 1.0)
            # 3: Relative Y distance to Zombie (-1.0 to 1.0)
            # 4: Relative X distance to Tree (-1.0 to 1.0)
            # 5: Relative Y distance to Tree (-1.0 to 1.0)
            # 6: Normalized HP of Agent (0.0 to 1.0)
            # 7: Normalized HP of Zombie (0.0 to 1.0)
            # 8: Binary flag (0.0 or 1.0) indicating if the agent has a stick
        low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(9,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Converts the current game state into a normalized 1D vector for the Neural Network.
        """
        # Normalize Agent position between -1.0 and 1.0
        norm_x = -(2.0 * (self.agent_pos[0] / self.max_x) - 1.0)
        norm_y = 2.0 * (self.agent_pos[1] / self.max_y) - 1.0
        
        # Normalize distance to Zombie between -1.0 and 1.0
        norm_zombie_dist_x = (self.agent_pos[0] - self.zombie_pos[0]) / self.max_x
        norm_zombie_dist_y = -(self.agent_pos[1] - self.zombie_pos[1]) / self.max_y
        
        # Normalize distance to Tree between -1.0 and 1.0
        norm_tree_dist_x = (self.agent_pos[0] - self.tree_pos[0]) / self.max_x
        norm_tree_dist_y = -(self.agent_pos[1] - self.tree_pos[1]) / self.max_y
        
        # Normalize HP (Max HP in V4 is 2)
        norm_agent_hp = max(0, self.agent_hp) / 2.0
        norm_zombie_hp = max(0, self.zombie_hp) / 2.0
        
        return np.array([
            norm_x, norm_y,
            norm_zombie_dist_x, norm_zombie_dist_y,
            norm_tree_dist_x, norm_tree_dist_y,
            norm_agent_hp,
            norm_zombie_hp,
            self.has_stick
            ], dtype=np.float32)

    def _get_info(self) -> dict:
        """Returns standard game variables for debugging or logging."""
        return {
            "agent_hp": self.agent_hp,
            "zombie_hp": self.zombie_hp,
            "has_stick": self.has_stick,
            "steps": self.current_step
        }

    def reset(self, seed=None, options=None):
        """Resets the game to the starting state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.recieved_stick_reward = False

        # Create an empty grid
        self.grid = np.zeros((self.size, self.size))

        # Place the agent in the middle
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.AGENT

        # Pick two random and opposite quadrants
        quadrants = self._get_quadrant_outer_zones()
        random_index = int(self.np_random.integers(0, 4))
        quadrant_A = quadrants[random_index]
        quadrant_B = quadrants[(random_index + 2) % 4]

        # Place the tree randomly in quadrant_A
        self.tree_pos = np.array([self.np_random.choice(quadrant_A)[0], self.np_random.choice(quadrant_A)[1]])
        self.grid[self.tree_pos[0], self.tree_pos[1]] = Entity.TREE

        # Place the zombie randomly in quadrant_B
        self.zombie_pos = np.array([self.np_random.choice(quadrant_B)[0], self.np_random.choice(quadrant_B)[1]])
        self.grid[self.zombie_pos[0], self.zombie_pos[1]] = Entity.ZOMBIE
        
        # Initialize the state variables
        self.agent_hp = 2
        self.zombie_hp = 2
        self.has_stick = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Applies an action, runs game logic, and returns the Gymnasium tuple."""
        self.current_step += 1

        # 1. Apply Action
        if action in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
            self._take_move_action(action)
        elif action == Action.ATTACK:
            self._take_attack_action()

        if self.logging:
            print(f"Agent took action {Action(action).name}")

        # 2. Run Enemy Logic
        self._run_game_logic()

        # 3. Calculate Reward
        reward = self._get_reward()

        # 4. Check Terminated / Truncated status
        terminated = bool(self.agent_hp <= 0 or self.zombie_hp <= 0)
        truncated = bool(self.current_step >= self.max_steps)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # --- Internal Game Logic Methods ---

    def _is_within_bounds(self, position: np.ndarray) -> bool:
        return (self.min_x <= position[0] <= self.max_x) and \
               (self.min_y <= position[1] <= self.max_y)

    def _is_next_to(self, entity: Entity) -> bool:
        for neighbor in self.NEIGHBORS:
            neighbor_pos = self.agent_pos + neighbor
            if self._is_within_bounds(neighbor_pos):
                if self.grid[neighbor_pos[0], neighbor_pos[1]] == entity:
                    return True
        return False

    def _get_quadrant_outer_zones(self):
        N = self.size
        C = N // 2  # Center index

        # Outer halfway boundaries
        top_rows = range(0, (C // 2) + 1)
        bottom_rows = range(C + (N - C) // 2, N)

        left_cols = range(0, (C // 2) + 1)
        right_cols = range(C + (N - C) // 2, N)

        # Generate list of (row, col) tuples for each quadrant
        return [
            [(r, c) for r in top_rows for c in left_cols],  # Top-left quadrant
            [(r, c) for r in top_rows for c in right_cols],  # Top-right quadrant
            [(r, c) for r in bottom_rows for c in right_cols],  # Bottom-right quadrant
            [(r, c) for r in bottom_rows for c in left_cols],  # Bottom-left quadrant
        ]

    def _take_move_action(self, action: int):
        target_x = min(max(self.agent_pos[0] + self.ACTIONS_TO_MOVE[action][0], self.min_x), self.max_x)
        target_y = min(max(self.agent_pos[1] + self.ACTIONS_TO_MOVE[action][1], self.min_y), self.max_y)

        # Check if the target position is blocked
        if self.grid[target_x, target_y] != Entity.EMPTY:
            return

        # Move the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.EMPTY  
        self.agent_pos = np.array([target_x, target_y])                 
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.AGENT  
    
    def _take_attack_action(self):
        if self._is_next_to(Entity.TREE):
            self.has_stick = True
        if self._is_next_to(Entity.ZOMBIE):
            damage = 2 if self.has_stick else 1
            self.zombie_hp -= damage

    def _run_game_logic(self):
        if self.zombie_hp > 0:
            # 1. Move towards the agent with 50% chance
            if not self._is_next_to(Entity.ZOMBIE) and self.np_random.random() < 0.5:
                dx = self.agent_pos[0] - self.zombie_pos[0]
                dy = self.agent_pos[1] - self.zombie_pos[1]

                target_x, target_y = self.zombie_pos[0], self.zombie_pos[1]

                # Pick randomly if both are available, otherwise pick whichever isn't 0
                use_x = bool(self.np_random.choice([True, False])) if (dx and dy) else bool(dx)
                if use_x:
                    target_x += 1 if dx > 0 else -1
                else:
                    target_y += 1 if dy > 0 else -1

                # Only move if the target cell is empty
                if self.grid[target_x, target_y] == Entity.EMPTY:
                    self.grid[self.zombie_pos[0], self.zombie_pos[1]] = Entity.EMPTY
                    self.zombie_pos = np.array([target_x, target_y])
                    self.grid[target_x, target_y] = Entity.ZOMBIE

            # 2. If the zombie is next to the agent, it attacks (even if it just moved)
            if self._is_next_to(Entity.ZOMBIE):
                self.agent_hp -= 1
                    

    def _get_reward(self) -> float:
        time_penalty = -0.1
        agent_dies_reward = -10.0 if self.agent_hp <= 0 else 0.0
        zombie_dies_reward = 10.0 if self.zombie_hp <= 0 else 0.0

        got_stick_reward = 0.0
        if self.has_stick and not self.recieved_stick_reward:
            got_stick_reward = 5.0
            self.recieved_stick_reward = True

        return float(time_penalty + got_stick_reward + agent_dies_reward + zombie_dies_reward)

    def render(self):
        """Standard Gymnasium render method."""
        if self.render_mode == "human":
            if self.window is None:
                from visualization import Visualization
                self.window = Visualization()
                self.window.show()
            
            # Determine if we should show a game over message
            status_message = None
            if self.agent_hp <= 0:
                status_message = "GAME OVER - YOU DIED!"
            elif self.zombie_hp <= 0:
                status_message = "VICTORY - ZOMBIE DEFEATED!"
            
            # Pass the environment AND the status message
            self.window.render(self, status_message=status_message)

    def close(self):
        """Cleans up the Pygame window when the environment is closed."""
        if self.window is not None:
            self.window.close()
            self.window = None
