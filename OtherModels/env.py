import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Dynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, state_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def main():
    dynamics = Dynamics(17,6,256,5)
    print(dynamics)

if __name__ == "__main__":
    main()
