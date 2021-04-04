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

generator_kwargs = dict(activation_function=torch.nn.ReLU)
discr, gener = train_gan(projection_input, plot=False)


## 

inputs = gener.generate_generator_input(1)
def func(x):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    torch.nn.modules.module.register_module_forward_hook(hook)
    gener(x)
    return tuple(activations)

#pdb.set_trace()
jac = get_jacobian(inputs=inputs, func=func)


## Compute rank for input --> layer for layers

ranks = list(map(get_rank, jac))
print("Ranks:", *ranks)


## Estimate distribution of ID of generator output

n_estimates = 100
rank_estimates = np.zeros([n_estimates, len(ranks)])
for ix, x in enumerate(
    tqdm(gener.generate_generator_input(n_estimates))
):
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

## In[7]:

inputs = gener.generate_generator_input(1)
func = lambda x: ProjectionFunction()(gener(x))
get_jacobian(inputs=inputs, func=func)

