#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from datagen import (
    get_parabolic_data,
    get_circle_data,
    get_plane_data,
)
from gan import train_gan
from twonn_dimension import twonn_dimension
from models import ProjectionFunction

SEED = np.random.randint(1000)
print(f"Running with seed {SEED}")
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluate(
    estimate_id,
    get_data,
):
    # Initialize dataframe
    xlabel = "Dataset size"
    ylabel = "Intrinsic dimension"
    plot_data = pd.DataFrame()

    # Loop data
    n_experiments = 100
    n_data_vec = np.rint(np.logspace(1, 3, 3)).astype(int)

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

get_jacobian = torch.autograd.functional.jacobian

def main():

    # Projection
    n_data = 100
    circle_input, _ = get_circle_data(n_data)
    plane_input, _ = get_plane_data(n_data)

    projection_input = torch.hstack([circle_input, plane_input])

    discr, gener = train_gan(projection_input)

    func = gener
    inputs = gener.generate_generator_input(1)
    jac = get_jacobian(func=func, inputs=inputs).squeeze()
    print("Generator output ID is ", torch.matrix_rank(jac))

    proj_func = ProjectionFunction()
    func = lambda x: proj_func(gener(x))
    jac = get_jacobian(func=func, inputs=inputs).squeeze()
    print("ANN output ID is ", torch.matrix_rank(jac))

    evaluate(
        estimate_id=twonn_dimension,
        get_data=lambda n: get_parabolic_data(n)[1],
    )

    plt.show()

if __name__=="__main__":
    main()
