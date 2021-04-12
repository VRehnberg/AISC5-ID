#!/usr/bin/env python
# coding: utf-8

## Imports:


import pdb
import torch
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#import seaborn as sns
from corner import corner
from tqdm import tqdm, trange

from datagen import *
from gan import train_gan
from twonn_dimension import twonn_dimension
from models import ProjectionFunction

SEED = np.random.randint(1000)
print(f"Running with seed {SEED}")
np.random.seed(SEED)
torch.manual_seed(SEED)

from IPython import display
get_ipython().run_line_magic('pdb', 'off')
get_ipython().run_line_magic('matplotlib', 'inline')


## Define functions


get_jacobian = torch.autograd.functional.jacobian

get_rank = lambda x: int(torch.matrix_rank(x.squeeze()))

## Train GAN

n_data = 100
circle_input, _ = get_circle_data(n_data)
plane_input, _ = get_plane_data(n_data)

projection_input = torch.hstack([circle_input, plane_input])

Activation = torch.nn.ReLU
Optimizer = torch.optim.Adam
generator_kwargs = dict(activation_function=Activation)
discr, gener = train_gan(
    projection_input,
    plot=False,
    Optimizer=Optimizer,
)


## 

inputs = gener.generate_generator_input(1)
def func(x):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    h = torch.nn.modules.module.register_module_forward_hook(hook)
    gener(x)
    h.remove()
    return tuple(activations)

def layer_names():
    names = []
    def hook(module, input, output):
        names.append(type(module).__name__)
    h = torch.nn.modules.module.register_module_forward_hook(hook)
    gener.generate_data(1)
    h.remove()
    return names

#pdb.set_trace()
jac = get_jacobian(inputs=inputs, func=func)


## Compute rank for input --> layer for layers

ranks = list(map(get_rank, jac))
print("\nRanks:", *ranks)


## Estimate distribution of ID of generator output

n_estimates = 1000
rank_estimates = np.zeros([n_estimates, len(ranks)])
all_inputs = gener.generate_generator_input(n_estimates)
for ix, x in enumerate(tqdm(all_inputs)):
    x = x.reshape(1, -1)
    jac = get_jacobian(inputs=x, func=func)
    rank_estimates[ix, :] =  list(map(get_rank, jac))

# Remove doublet of last layer
rank_estimates = rank_estimates[:, :-1]

# Plot lines
fig, ax = plt.subplots()
x = np.tile(
    np.arange(rank_estimates.shape[1]),
    [rank_estimates.shape[0], 1],
)
plot_data = np.dstack([x, rank_estimates])
lines = LineCollection(
    plot_data,
    colors="b",
    alpha=0.01,
)
ax.add_collection(lines)

ax.set_xlim([0, x.shape[1]])
ax.set_ylim([0, rank_estimates.max()])

file_specifier = f"_{Optimizer.__name__}_{Activation.__name__}"
fig.tight_layout()
fig.savefig(f"Figures/lines{file_specifier}.pdf", bbox_inches="tight")

# Distribution
corner_cols = np.arange(1, 7)
corner_data = rank_estimates[:, corner_cols]
corner_labels = np.array(layer_names())[corner_cols]
fig = corner(
    corner_data,
    labels=corner_labels,
    show_titles=True,
)

fig.tight_layout()
fig.savefig(f"Figures/corner{file_specifier}.pdf", bbox_inches="tight")

## In[7]:

inputs = gener.generate_generator_input(1)
func = lambda x: ProjectionFunction()(gener(x))
get_jacobian(inputs=inputs, func=func)

