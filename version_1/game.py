import numpy as np

class Map():

    # Entity types
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2

    # Constants
    NEIGHBORS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]

    def __init__(self, size = 5):
        # Map is always a square with odd size
        self.min_x = 0
        self.max_x = size - 1
        self.min_y = 0
        self.max_y = size - 1

        # Initialize the grid
        self.grid = np.zeros((size, size))

        # Initialize the resource
        self.resource_pos = np.array([size - 2, size - 1])
        self.grid[self.resource_pos[0], self.resource_pos[1]] = self.RESOURCE

        # Initialize the agent
        self.agent_pos = np.array([size // 2, size // 2]) # Center
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        # Define the actions
        self.actions_to_move = {
            "up": self.NEIGHBORS[0],
            "right": self.NEIGHBORS[1],
            "down": self.NEIGHBORS[2],
            "left": self.NEIGHBORS[3]
        }

    def is_within_bounds(self, position):
        return position[0] >= self.min_x and position[0] <= self.max_x and position[1] >= self.min_y and position[1] <= self.max_y

    def is_next_to(self, entity):
        for neighbor in self.NEIGHBORS:
            neighbor_pos = self.agent_pos + neighbor
            if self.is_within_bounds(neighbor_pos):
                if self.grid[neighbor_pos[0], neighbor_pos[1]] == entity:
                    return True

    def action(self, action):
        # Find the new position, stop at the edge
        new_pos_x = min(max(self.agent_pos[0] + self.actions_to_move[action][0], self.min_x), self.max_x)
        new_pos_y = min(max(self.agent_pos[1] + self.actions_to_move[action][1], self.min_y), self.max_y)

        # Update the grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.EMPTY
        self.grid[new_pos_x, new_pos_y] = self.AGENT

        # Update the agent position
        self.agent_pos = np.array([new_pos_x, new_pos_y])

        # Log
        print("Agent took action", action)

        return self.get_reward()
    
    def get_reward(self):
        # Small penatly for taking an action
        time_penalty = -0.1
        position_reward = 0

        # Reward the agent for finding a resource
        if self.is_next_to(self.RESOURCE):
            position_reward = 1
        
        # Calculate the total reward
        total_reward = time_penalty + position_reward

        # Log
        print("Agent got reward", total_reward)
        
        return total_reward
    
    def __str__(self):
        return "Map:\n" + str(self.grid) + "\n"