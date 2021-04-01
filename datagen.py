import math
import torch

def get_parabolic_data(
    n_data,
    eps=0.0,
):
    x = 3 * torch.rand(n_data) - 1
    y = x**2 + eps * torch.randn(*x.shape)

    return x.reshape(-1, 1), y.reshape(-1, 1)

def get_circle_data(
    n_data,
):

    r = 1.0
    ang = 2.0 * math.pi * torch.rand(n_data)
    x = r * torch.cos(ang)
    y = r * torch.sin(ang)

    input = torch.vstack([x, y]).T
    output = ang.reshape(-1, 1)
    return input, output


def get_plane_data(
    n_data,
):

    x = torch.rand(n_data)
    y = torch.rand(n_data)
    z = 2.0 * x + y

    input = torch.vstack([x, y]).T
    output = z.reshape(-1, 1)
    return input, output
