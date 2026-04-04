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

class Map:
    # The 4 cardinal directions
    NEIGHBORS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    
    ACTIONS_TO_MOVE = {
        Action.UP: NEIGHBORS[0],
        Action.RIGHT: NEIGHBORS[1],
        Action.DOWN: NEIGHBORS[2],
        Action.LEFT: NEIGHBORS[3]
    }

    def __init__(self, size: int = 5, logging: bool = False):
        # Define the game map
        self.size = size
        self.min_x = 0
        self.max_x = size - 1
        self.min_y = 0
        self.max_y = size - 1
        self.logging = logging

        # Store if the agent got the stick reward
        self.recieved_stick_reward = False

        # Define the state size
        self.num_positions = self.size * self.size
        self.num_stick_states = 2 # False or True
        self.num_agent_hp = 3     # 0, 1, 2
        self.num_zombie_hp = 3    # 0, 1, 2

        # Initialize the game
        self.reset()

    def is_within_bounds(self, position: np.ndarray) -> bool:
        return (self.min_x <= position[0] <= self.max_x) and \
               (self.min_y <= position[1] <= self.max_y)

    def is_next_to(self, entity: Entity) -> bool:
        for neighbor in self.NEIGHBORS:
            neighbor_pos = self.agent_pos + neighbor
            if self.is_within_bounds(neighbor_pos):
                if self.grid[neighbor_pos[0], neighbor_pos[1]] == entity:
                    return True
        return False
                
    def is_on_top_of(self, entity: Entity) -> bool:
        return self.grid[self.agent_pos[0], self.agent_pos[1]] == entity
                
    def get_state(self) -> int:
        # Get the current state
        pos = self.agent_pos[0] * self.size + self.agent_pos[1]
        stick = 1 if self.has_stick else 0
        a_hp = self.agent_hp
        z_hp = self.zombie_hp

        # Convert state to a single number
        state_id = pos
        state_id += stick * (self.num_positions)
        state_id += a_hp  * (self.num_positions * self.num_stick_states)
        state_id += z_hp  * (self.num_positions * self.num_stick_states * self.num_agent_hp)
        
        return int(state_id)

    def take_action(self, action: int) -> float:
        # If the action is a move
        if action in (Action.UP.value, Action.RIGHT.value, Action.DOWN.value, Action.LEFT.value):
            self.take_move_action(action)

        # If the action is an attack
        elif action == Action.ATTACK.value:
            self.take_attack_action()

        if self.logging:
            print(f"Agent took action {Action(action).name}")
            print(self.grid)

        # Run game logic
        self.run_game_logic()

        # Calculate and return reward
        return self.get_reward()
    
    def take_move_action(self, action):
        # Calculate where the agent will go
        target_x = min(max(self.agent_pos[0] + self.ACTIONS_TO_MOVE[action][0], self.min_x), self.max_x)
        target_y = min(max(self.agent_pos[1] + self.ACTIONS_TO_MOVE[action][1], self.min_y), self.max_y)

        # Check if the target position is blocked
        if self.grid[target_x, target_y] != Entity.EMPTY:
            target_x, target_y = self.agent_pos[0], self.agent_pos[1]

        # Move the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.EMPTY  # Set previous position to empty
        self.agent_pos = np.array([target_x, target_y])                 # Save new position
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.AGENT  # Set new position to agent
    
    def take_attack_action(self):
        # If next to the tree, get stick
        if self.is_next_to(Entity.TREE):
            self.has_stick = True

        # If next to the zombie, attack
        elif self.is_next_to(Entity.ZOMBIE):
            damage = 2 if self.has_stick else 1
            self.zombie_hp -= damage

    def run_game_logic(self):
        # If the zombie is alive, and next to the agent, it attacks
        if  self.zombie_hp > 0 and self.is_next_to(Entity.ZOMBIE):
            self.agent_hp -= 1
    
    def get_reward(self) -> float:
        # Small negative reward per action
        time_penalty = -0.1

        # Negative reward if the agent dies
        agent_dies_reward = -10 if self.agent_hp <= 0 else 0

        # Positive reward if the zombie dies
        zombie_dies_reward = 10 if self.zombie_hp <= 0 else 0

        # Positive reward if the agent gets the stick
        if self.has_stick and not self.recieved_stick_reward:
            got_stick_reward = 5
            self.recieved_stick_reward = True
        else:
            got_stick_reward = 0

        total_reward = time_penalty + got_stick_reward + agent_dies_reward + zombie_dies_reward

        if self.logging:
            print(f"Agent got reward {total_reward}\n")
        
        return total_reward
    
    def is_terminal_state(self) -> bool:
        return self.agent_hp <= 0 or self.zombie_hp <= 0
    
    def reset(self) -> None:
        # Create an empty grid
        self.grid = np.zeros((self.size, self.size))

        # Add the agent in the middle
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.AGENT

        # Add the tree in the bottom right
        self.tree_post = np.array([self.size - 1, self.size - 1])
        self.grid[self.tree_post[0], self.tree_post[1]] = Entity.TREE

        # Add the zombie in the top middle
        self.zombie_pos = np.array([0, self.size // 2])
        self.grid[self.zombie_pos[0], self.zombie_pos[1]] = Entity.ZOMBIE

        # Initalize the state variables
        self.agent_hp = 2
        self.zombie_hp = 2
        self.has_stick = False