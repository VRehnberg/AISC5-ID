import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange

from twonn_dimension import twonn_dimension

def get_parabolic_data(
    n_data,
    eps=0.0,
    random_state=np.random,
):
    x = 3 * random_state.rand(100) - 1
    y = x**2 + eps * random_state.randn(*x.shape)

    data = np.vstack([x, y]).T

    return data

def evaluate(
    estimate_id,
    get_data,
):
    # Initialize dataframe
    xlabel = "Dataset size"
    ylabel = "Intrinsic dimension"
    plot_data = pd.DataFrame()

    # Loop data
    n_experiments = 1000
    n_data_vec = np.rint(np.logspace(1, 5, 5)).astype(int)

    for i_n_data, n_data in enumerate(tqdm(n_data_vec)):

        # Initialize values to put in dataframe
        n_data_dist = np.log10(np.full(n_experiments, n_data))
        id_dist = np.zeros(n_experiments)

        for i_exp in trange(n_experiments, desc=f"{n_data}"):

            # Get data and estimate intrinsic dimension
            data = get_data(n_data)
            id_dist[i_exp] = estimate_id(data)

        # Add data to dataframe
        plot_data = plot_data.append(pd.DataFrame({
            xlabel: n_data_dist,
            ylabel: id_dist,
        }))


    # Plot distributions of ID
    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x=xlabel,
        y=ylabel,
        data=plot_data,
    )

    xticks = np.log10(n_data_vec)
    ax.set_xticks(xticks-1)
    ax.set_xticklabels([f"$10^{int(x):d}$" for x in xticks])

    return fig, ax

def main():
    
    evaluate(
        estimate_id=twonn_dimension,
        get_data=get_parabolic_data,
    )

    plt.show()

if __name__=="__main__":
    main()
