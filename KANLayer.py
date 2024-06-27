# Python libraries
from typing import List

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# User-defined libraries
from bspline import compute_bspline_wrapper


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
        self.residual_std = residual_std
        self.device = device


        # Define control points
        spacing = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, device=device) * spacing + grid_range[0]
        # Generate (out, in) copies of equally spaced points on [a, b]
        grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()
        self.register_buffer("grid", grid)

        # Residual activation functions
        self.residual_fn = F.silu
        self.residual_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))  # w_b in paper

        # Define univariate function (splines in original KAN)
        self.univariate_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))  # w_s in paper
        self.univarate_fn = compute_bspline_wrapper(self.grid, self.spline_order, self.device)

        # Spline parameters
        self.coef = torch.nn.Parameter(torch.Tensor(out_dim, in_dim, grid_size + spline_order))
        
        self._initialization()

    def _initialization(self):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.normal_(self.residual_weight, mean=0.0, std=self.residual_std)
        nn.init.ones_(self.univariate_weight)
        nn.init.xavier_normal_(self.coef)

    @torch.no_grad
    def grid_extension(self, x: torch.Tensor, new_sz: int):
        """
        Increase granularity of B-spline by increasing the number of grid points. We
        ensure the B-spline over the finer grid maintains the shape of the original through
        least-squares.
        """
        pass

    @torch.no_grad
    def control_point_update(self, x: torch.Tensor):
        """
        Perturb the control points on the grid to be non-uniform and adjust the
        B-spline coefficients accordingly.
        """
        pass

    def forward(self, x: torch.Tensor):
        """
        Forward pass of KAN. x is expected to be of shape (batch_size, input_size) where
        input_size is the number of input scalars.
        """
        "Spline grids are updated during the forward pass."

        # Broadcast [batch_size x in_dim] to [batch_size x out_dim x in_dim]
        # x = x[:, None, :].expand(-1, self.out_dim, -1)

        bases = self.univarate_fn(x)
        spline = torch.sum(bases * self.coef[None, ...], dim=-1)

        # Form the batch of matrices phi(x) of shape [batch_size x out_dim x in_dim]
        phi = self.residual_weight * self.residual_fn(x[:, None, :]) + \
              self.univariate_weight * spline
        
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
    grid_range = [-1.,1]

    x = torch.ones(bsz, in_dim) / 0.8

    spacing = (grid_range[1] - grid_range[0]) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, device=device) * spacing + grid_range[0]
    # Generate (out, in) copies of equally spaced points on [a, b]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()

    print('x', x)
    print('grid', grid)
    layer = KANLayer(in_dim, out_dim, grid_size, spline_order, device, grid_range=grid_range)
    
    y = layer(x)

    print(y.shape)



