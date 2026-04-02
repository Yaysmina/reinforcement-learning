import numpy as np

class Map():

    # Entity types
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2
    WALL = 3

    # Constants
    NEIGHBORS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    ACTIONS_TO_MOVE = {0: NEIGHBORS[0], 1: NEIGHBORS[1], 2: NEIGHBORS[2], 3: NEIGHBORS[3]}
    ACTIONS_TO_NAME = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


    def __init__(self, size = 5, walls = 0):
        # Map is always a square with odd size
        self.size = size
        self.min_x = 0
        self.max_x = size - 1
        self.min_y = 0
        self.max_y = size - 1

        # Walls are off by default
        self.walls_count = walls
        self.wall_coords = []

        # Generate static walls
        self.generate_static_walls()
        
        # Logging is off by default
        self.logging = False

        # Initialize the map
        self.reset()
    
    def generate_static_walls(self):
        # We need a temporary grid to make sure we don't place walls 
        # where the agent or resource starts.
        temp_grid = np.zeros((self.size, self.size))
        agent_start = [self.size // 2, self.size // 2]
        resource_start = [self.size - 1, self.size - 1]
        
        placed_walls = 0
        while placed_walls < self.walls_count:
            # Use self.size to include the last row/column
            wall_pos = np.random.randint(0, self.size, size=2)
            
            # Don't place a wall on Agent, Resource, or an existing Wall
            if not (np.array_equal(wall_pos, agent_start) or 
                    np.array_equal(wall_pos, resource_start) or 
                    temp_grid[wall_pos[0], wall_pos[1]] == self.WALL):
                
                self.wall_coords.append(wall_pos)
                temp_grid[wall_pos[0], wall_pos[1]] = self.WALL
                placed_walls += 1

    def is_within_bounds(self, position):
        return position[0] >= self.min_x and position[0] <= self.max_x and position[1] >= self.min_y and position[1] <= self.max_y

    def is_next_to(self, entity):
        for neighbor in self.NEIGHBORS:
            neighbor_pos = self.agent_pos + neighbor
            if self.is_within_bounds(neighbor_pos):
                if self.grid[neighbor_pos[0], neighbor_pos[1]] == entity:
                    return True
                
    def is_on_top_of(self, entity):
        return self.grid[self.agent_pos[0], self.agent_pos[1]] == entity
                
    def get_state(self):
        return self.size * self.agent_pos[0] + self.agent_pos[1]

    def take_action(self, action):
            # Calculate where the agent wants to go
            target_x = min(max(self.agent_pos[0] + self.ACTIONS_TO_MOVE[action][0], self.min_x), self.max_x)
            target_y = min(max(self.agent_pos[1] + self.ACTIONS_TO_MOVE[action][1], self.min_y), self.max_y)

            # Check if the target is empty
            if self.grid[target_x, target_y] != self.EMPTY:
                target_x, target_y = self.agent_pos[0], self.agent_pos[1]

            # Update the grid 
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.EMPTY
            
            # Update position
            self.agent_pos = np.array([target_x, target_y])
            
            # Place agent in new spot
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

            if self.logging:
                print("Agent took action", self.ACTIONS_TO_NAME[action])
                print(self.grid)

            return self.get_reward()
    
    def get_reward(self):
        # Small penatly for taking an action
        time_penalty = -0.01
        position_reward = 0

        # Reward the agent for finding a resource
        if self.is_next_to(self.RESOURCE):
            position_reward = 1
        
        # Calculate the total reward
        total_reward = time_penalty + position_reward

        # Log
        if self.logging:
            print("Agent got reward", total_reward)
            print()
            print("----------------------")
            print()
        
        return total_reward
    
    def is_terminal_state(self):
        return self.is_next_to(self.RESOURCE)
    
    def reset(self):
        # Initialize the grid with zeros
        self.grid = np.zeros((self.size, self.size))

        # Re-place the walls from our static list
        for wall in self.wall_coords:
            self.grid[wall[0], wall[1]] = self.WALL

        # Initialize the resource
        self.resource_pos = np.array([self.size - 1, self.size - 1])
        self.grid[self.resource_pos[0], self.resource_pos[1]] = self.RESOURCE

        # Initialize the agent
        # self.agent_pos = np.array([self.size // 2, self.size // 2])
        self.agent_pos = np.array([0, 0])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT