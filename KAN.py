# Python libraries
from typing import List
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
        self.layers = []

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
    
    def initialize_from_KAN(self):
        pass


if __name__ == "__main__":
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = KANConfig()
    layers = [2, 5, 1]
    model = KAN(layer_widths=layers, config=config)

    bsz = 4
    x = torch.ones(bsz, 2) / 0.8

    print(model(x))
