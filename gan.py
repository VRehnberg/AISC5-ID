import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def prepare_animation(bar_container):

    def animate(frame_number):
        # simulate new data coming in
        data = np.random.randn(1000)
        n, _ = np.histogram(data, HIST_BINS)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        return bar_container.patches
    return animate

class Generator(nn.Module):

    def __init__(
        self,
        generator_input_dim,
        generator_output_dim,
        hidden_output_dims=[8, 8, 8],
        Activation=nn.ReLU,
    ):
        super().__init__()

        self.input_dim = generator_input_dim

        # TODO experiment architecture
        hidden_input_dims = [
            generator_input_dim,
            *hidden_output_dims[:-1],
        ]
        self.hidden_layers = [
            layer
            for hd_in, hd_out in zip(
                hidden_input_dims,
                hidden_output_dims,
            )
            for layer in [
                nn.Linear(hd_in, hd_out),
                Activation(),
            ]
        ]
        self.output_layer = nn.Linear(
            hidden_output_dims[-1],
            generator_output_dim,
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def generate_generator_input(self, n_data):
        return torch.randn(n_data, self.input_dim)

    def generate_x(self, gener_input):
        gener_output = self.forward(gener_input)
        return gener_output

    def generate_data(self, n_data):
        gener_input = self.generate_generator_input(n_data)
        gener_x = self.generate_x(gener_input)
        return gener_input, gener_x


# TODO give discriminator access to activations?
class Discriminator(nn.Module):
    
    def __init__(
        self,
        input_dim,
        hidden_output_dims=[8, 8, 8],
        Activation=nn.ReLU,
    ):
        super().__init__()

        # TODO experiment architecture
        hidden_input_dims = [
            input_dim,
            *hidden_output_dims[:-1],
        ]
        self.hidden_layers = [
            layer
            for hd_in, hd_out in zip(
                hidden_input_dims,
                hidden_output_dims,
            )
            for layer in [
                nn.Linear(hd_in, hd_out),
                Activation(),
            ]
        ]
        self.output_layer = nn.Linear(hidden_output_dims[-1], 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.output_activation(x)

def train_gan(
    data_input,
    n_epochs=1000,
    plot=False,
    Optimizer=optim.Adam,
    generator_kwargs={},
    discriminator_kwargs={},
):

    # Load data
    n_data, data_input_dim = data_input.shape

    # Initialize generator
    gener_input_dim = 10
    gener_output_dim = data_input_dim
      
    gener = Generator(
        generator_input_dim=gener_input_dim,
        generator_output_dim=gener_output_dim,
        **generator_kwargs,
    )

    # Initialize discriminator
    discr_input_dim = gener_output_dim
    discr = Discriminator(discr_input_dim, **discriminator_kwargs)

    # Define loss and optimizer TODO experiment
    loss_function = nn.BCELoss()
    lr = 0.02
    opt = Optimizer([
        {"params" : gener.parameters(), "lr" : -lr},
        {"params" : discr.parameters(), "lr" : lr},
    ])

    # Prepare histogram
    previous_patches = None
    def init_plot():
        nonlocal previous_patches

        fig, ax = plt.subplots()
        hist_kwargs = dict(
            bins=100,
            range=(-1, 2),
            histtype="step",
            density=True,
            cumulative=True,
        ) 
        ax.hist(
            data_input.detach().numpy(),
            **hist_kwargs,
            ec="blue",
            label="True",
        )
        plot_gener_input, fake_x = gener.generate_data(n_data)

        previous_patches = []

    def update_plot():
        '''Help function to update plot.'''
        nonlocal plot
        if plot:
            return

        nonlocal previous_patches
        nonlocal gener
        nonlocal discr

        gener.eval()
        discr.eval()

        # Get generator histogram
        for previous_patch in previous_patches:
            previous_patch.remove()
        previous_patches = []

        with torch.no_grad():
            fake_x = gener.generate_x(plot_gener_input)
        fake_x = fake_x.detach().numpy()
        _, _, patches = ax.hist(
            fake_x,
            **hist_kwargs,
            ec="red",
            label="Fake",
        )
        previous_patches.extend(patches)

        # Plot discriminator
        x = torch.linspace(-1, 2, 100).reshape(-1, 1)
        discr_output = discr(x).detach().numpy()
        line2ds = ax.plot(x, discr_output, "-", c="yellow")
        previous_patches.extend(line2ds)

        # Redraw canvas
        fig.canvas.draw()
        fig.canvas.flush_events()


    if plot:
        init_plot()
        update_plot()

    pbar = tqdm(total=n_epochs)
    for i_epoch in range(n_epochs):

        # Prepare for training
        gener.train()
        gener.zero_grad()
        discr.train()
        discr.zero_grad()

        # Generate artificial input data
        gener_input, gener_output = gener.generate_data(n_data)

        # Build discriminator data
        discr_input = torch.vstack([data_input, gener_output])
        discr_label = torch.vstack([
            torch.ones(n_data, 1),
            torch.zeros(n_data, 1),
        ])

        # Calculate loss
        discr_predict = discr(discr_input)
        loss = loss_function(discr_predict, discr_label)

        pbar.update(1)
        pbar.set_description(f"Loss {loss:.4f}")

        # Update parameter values
        loss.backward(retain_graph=True)
        opt.step()

        if (i_epoch % 1) == 0:
            if plot:
                plt.pause(0.0001)
                update_plot()

    pbar.close()

    return discr, gener
