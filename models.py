import torch
from torch import nn

class SimpleModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class ProjectionFunction(nn.Module):

    def forward(self, input_data):
        x_circle, y_circle, x_plane, y_plane = torch.split(
            input_data,
            1,
            dim=1,
        )

        angle = torch.sign(y_circle) * torch.acos(x_circle)
        line = 2 * x_plane + y_plane

        return torch.vstack([angle, line]).T
