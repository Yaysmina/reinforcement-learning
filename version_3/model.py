import torch
from torch import nn
from environment import GridMuckEnvV2
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    # Create an environment
    env = GridMuckEnvV2(size=5)

    # Define model shape based on environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create model with random starting state
    model = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)

    # Run the model
    q_values = model(state)

    # Print model and q_values
    print(f"Model: {model}")
    print(f"Q-Values: {q_values}")