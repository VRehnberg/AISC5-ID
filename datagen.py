import torch

def get_parabolic_data(
    n_data,
    eps=0.0,
):
    x = 3 * torch.rand(n_data) - 1
    y = x**2 + eps * torch.randn(*x.shape)

    data = torch.vstack([x, y]).T

    return data
