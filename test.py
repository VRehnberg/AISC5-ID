import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from twonn_dimension import twonn_dimension

def get_parabolic_data(
    n_data,
    eps=0.01,
    random_state=np.random,
):
    x = 3 * random_state.rand(100) - 1
    y = x**2 + eps * random_state.randn(*x.shape)

    data = np.vstack([x, y]).T

    return data


def main():
    
    n_experiments = 100
    n_data_vec = np.rint(np.logspace(1, 5, 17)).astype(int)

    id_mean = np.zeros_like(n_data_vec)
    id_95ci = np.zeros([2, id_mean.size])

    for i_n_data, n_data in enumerate(tqdm(n_data_vec)):

        id_tmp = np.zeros(n_experiments)

        for i_exp in range(n_experiments):

            data = get_parabolic_data(n_data)
            id_tmp[i_exp] = twonn_dimension(data)

        id_mean[i_n_data] = np.mean(id_tmp)
        id_95ci[:, i_n_data] = np.quantile(id_tmp, [0.025, 0.975])

    fig, ax = plt.subplots()
    ax.errorbar(
        x=n_data_vec,
        y=id_mean,
        yerr=id_95ci,
        fmt='.-',
        capsize=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("\#data points")
    ax.set_ylabel("ID estimate")

    plt.show()

if __name__=="__main__":
    main()
