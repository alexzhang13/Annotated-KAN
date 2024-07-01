# Python libraries
from typing import List, Self
import random

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# User-defined libraries
from KANLayer import KANLayer


class KANConfig:
    """
    Configuration struct to define a standard KAN.
    """

    residual_std = 0.1
    grid_size = 5
    spline_order = 3
    grid_range = [-1.0, 1.0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KAN(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.
    """

    def __init__(
        self,
        layer_widths: List[int],
        config: KANConfig,
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_widths = layer_widths
        self.config = config

        in_widths = layer_widths[:-1]
        out_widths = layer_widths[1:]

        for in_dim, out_dim in zip(in_widths, out_widths):
            self.layers.append(
                KANLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=config.grid_size,
                    spline_order=config.spline_order,
                    device=config.device,
                    residual_std=config.residual_std,
                    grid_range=config.grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Standard forward pass sequentially across each layer.
        """
        for layer in self.layers:
            x = layer(x)

        return x

    @torch.no_grad
    def fix_symbolic(self, layer: int, in_index: int, out_index: int, fn):
        """
        For layer {layer}, activation {in_index, out_index}, fix the output
        to the function fn. This is grossly inefficient, but works.
        """

    @torch.no_grad
    def prune(self, x: torch.Tensor, mag_threshold: float = 0.01):
        """
        Prune (mask) a node in a KAN layer if the normalized activation
        incoming or outgoing are lower than mag_threshold.
        """
        # Collect activations and cache
        self.forward(x)

        # Can't prune at last layer
        for l_idx in range(len(self.layers) - 1):
            # Average over the batch and take the abs of all edges
            in_mags = torch.abs(torch.mean(self.layers[l_idx].activations, dim=0))

            # (in_dim, out_dim), average over out_dim
            in_score = torch.max(in_mags, dim=-1)[0]

            # Average over the batch and take the abs of all edges
            out_mags = torch.abs(torch.mean(self.layers[l_idx + 1].activations, dim=0))

            # (in_dim, out_dim), average over out_dim
            out_score = torch.max(out_mags, dim=0)[0]

            # Check for input, output (normalized) activations > mag_threshold
            active_neurons = (in_score > mag_threshold) * (out_score > mag_threshold)
            inactive_neurons_indices = (active_neurons == 0).nonzero()

            # Mask all relevant activations
            self.layers[l_idx + 1].activation_mask[:, inactive_neurons_indices] = 0
            self.layers[l_idx].activation_mask[inactive_neurons_indices, :] = 0


    @torch.no_grad
    def grid_extension(self, x: torch.Tensor, new_grid_size: int):
        """
        Increase granularity of B-spline by changing the grid size
        in the B-spline computation to be new_grid_size.
        """
        self.forward(x)
        for l_idx in range(len(self.layers)):
            self.layers[l_idx].grid_extension(self.layers[l_idx].inp, new_grid_size)
        self.config.grid_size = new_grid_size



if __name__ == "__main__":
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = KANConfig()
    layers = [2, 5, 5, 5, 1]
    model = KAN(layer_widths=layers, config=config)

    bsz = 4
    x = torch.ones(bsz, 2) / 0.8

    model.prune(x)
    # model.grid_extension(x, new_grid_size=25)
