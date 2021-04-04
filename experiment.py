#!/usr/bin/env python
# coding: utf-8

## In[1]:


import pdb
import torch
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
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


## In[2]:


def get_jacobian(inputs, function, create_graph=False):
    jac = torch.autograd.functional.jacobian(
        func=function,
        inputs=inputs,
        create_graph=create_graph,
    )
    return jac


## In[3]:


# Projection
n_data = 100
circle_input, _ = get_circle_data(n_data)
plane_input, _ = get_plane_data(n_data)

projection_input = torch.hstack([circle_input, plane_input])

discr, gener = train_gan(projection_input)


## In[4]:


inputs = gener.generate_generator_input(1)
def func(x):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    torch.nn.modules.module.register_module_forward_hook(hook)
    gener(x)
    return tuple(activations)

#pdb.set_trace()
jac = get_jacobian(inputs, func)


## In[5]:

get_rank = lambda x: int(torch.matrix_rank(x.squeeze()))
ranks = list(map(get_rank, jac))
print("Ranks:", *ranks)


## In[6]:


inputs = gener.generate_generator_input(1)
func = lambda x: ProjectionFunction()(gener(x))
get_jacobian(inputs, func)

