import numpy as np
from enum import IntEnum

class Entity(IntEnum):
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

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
        self.size = size
        self.min_x = 0
        self.max_x = size - 1
        self.min_y = 0
        self.max_y = size - 1
        self.logging = logging
        
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
                
    def get_state(self) -> np.ndarray:
        position_state = self.agent_pos[0] * self.size + self.agent_pos[1]

        return np.array([position_state])

    def take_action(self, action: int) -> float:
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

        if self.logging:
            print(f"Agent took action {Action(action).name}")
            print(self.grid)

        return self.get_reward()
    
    def get_reward(self) -> float:
        time_penalty = -0.1
        position_reward = 10 if self.is_next_to(Entity.RESOURCE) else 0

        total_reward = time_penalty + position_reward

        if self.logging:
            print(f"Agent got reward {total_reward}\n")
            print("----------------------\n")
        
        return total_reward
    
    def is_terminal_state(self) -> bool:
        return self.is_next_to(Entity.RESOURCE)
    
    def reset(self) -> None:
        # Create an empty grid
        self.grid = np.zeros((self.size, self.size))

        # Add the resource
        self.resource_pos = np.array([self.size - 1, self.size - 1])
        self.grid[self.resource_pos[0], self.resource_pos[1]] = Entity.RESOURCE

        # Add the agent
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entity.AGENT