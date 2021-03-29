import numpy as np

def get_parabolic_data(
    n_data,
    eps=0.0,
    random_state=np.random,
):
    x = 3 * random_state.rand(100) - 1
    y = x**2 + eps * random_state.randn(*x.shape)

    data = np.vstack([x, y]).T

    return data
