# Python libraries
from typing import List

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# User-defined libraries
from bspline import compute_bspline_wrapper


def generate_control_points(
    low_bound: float,
    up_bound: float,
    spline_order: int,
    grid_size: int,
    device: torch.device,
):
    """ 
    Generate a vector of {grid_size} equally spaced points in the interval [low_bound, up_bound].
    To account for B-splines of order k, using the same spacing, generate an additional
    k points on each side of the interval. See 2.4 in original paper for details.
    """
    spacing = (up_bound - low_bound) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, device=device)
    grid = grid * spacing + low_bound

    return grid


class KANActivation(nn.Module):
    """
    Defines a KAN Activation layer that computes the spline(x) logic
    described in the original paper.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spline_order: int,
        grid_size: int,
        device: torch.device,
        grid_range: List[float],
    ):
        super(KANActivation, self).__init__()
        # Define control points
        grid = generate_control_points(
            grid_range[0], grid_range[1], spline_order, grid_size, device
        )
        # Generate (out, in) copies of equally spaced points on [a, b]
        grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()
        self.register_buffer("grid", grid)

        # Define the univariate B-spline function
        self.univarate_fn = compute_bspline_wrapper(
            self.grid, spline_order, device
        )

        # Spline parameters
        self.coef = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim, grid_size + spline_order)
        )

    def _initialization(self):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.xavier_normal_(self.coef)

    def forward(self, x: torch.Tensor):
        """
        Compute and evaluate the learnable activation functions
        applied to a batch of inputs of size in_dim each.
        """
        # Broadcast [batch_size x in_dim] to [batch_size x out_dim x in_dim]
        # x = x[:, None, :].expand(-1, self.out_dim, -1)

        bases = self.univarate_fn(x)
        postacts = bases * self.coef[None, ...]
        spline = torch.sum(postacts, dim=-1)

        return spline


class KANLayer(nn.Module):
    "Defines a KAN layer from in_dim variables to out_dim variables."

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        spline_order: int,
        device: torch.device,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device

        # Residual activation functions
        self.residual_fn = F.silu
        self.residual_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_b in paper

        # Define univariate function (splines in original KAN)
        self.activation_fn = KANActivation(
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            device,
            grid_range,
        )
        self.univariate_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_s in paper

        # Cache for regularization
        self.inp = torch.empty(0)
        self.activations = torch.empty(0)
        self.l1_activations = torch.empty(0)

        self._initialization(residual_std)

    def _initialization(self, residual_std):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.normal_(self.residual_weight, mean=0.0, std=residual_std)
        nn.init.ones_(self.univariate_weight)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of KAN. x is expected to be of shape (batch_size, input_size) where
        input_size is the number of input scalars.

        Stores the activations needed for computing the L1 regularization and
        entropy regularization terms.

        Returns the output of the KAN operation.
        """
        
        spline = self.activation_fn(x)

        # Form the batch of matrices phi(x) of shape [batch_size x out_dim x in_dim]
        phi = (
            self.residual_weight * self.residual_fn(x[:, None, :])
            + self.univariate_weight * spline
        ) 

        # Cache activations of training
        if True:  # self.training:
            self.inp = x
            self.activations = phi
            self.l1_activations = torch.sum(torch.mean(torch.abs(phi), dim=0))

        # Really inefficient matmul
        out = torch.sum(phi, dim=-1)

        return out


if __name__ == "__main__":
    print("KAN Layer Unit Tests")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bsz = 2
    spline_order = 3
    in_dim = 5
    out_dim = 7
    grid_size = 11
    grid_range = [-1.0, 1]

    x = torch.ones(bsz, in_dim) / 0.8

    spacing = (grid_range[1] - grid_range[0]) / grid_size
    grid = (
        torch.arange(-spline_order, grid_size + spline_order + 1, device=device)
        * spacing
        + grid_range[0]
    )
    # Generate (out, in) copies of equally spaced points on [a, b]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()

    print("x", x)
    print("grid", grid)
    layer = KANLayer(
        in_dim, out_dim, grid_size, spline_order, device, grid_range=grid_range
    )

    y = layer(x)

    print(y.shape)
