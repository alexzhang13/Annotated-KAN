# Python libraries
from typing import List, Self, Callable

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# User-defined libraries
from bspline import compute_bspline


def generate_control_points(
    low_bound: float,
    up_bound: float,
    in_dim: int,
    out_dim: int,
    spline_order: int,
    grid_size: int,
    device: torch.device,
):
    """
    Generate a vector of {grid_size} equally spaced points in the interval [low_bound, up_bound] and broadcast (out_dim, in_dim) copies.
    To account for B-splines of order k, using the same spacing, generate an additional
    k points on each side of the interval. See 2.4 in original paper for details.
    """

    # vector of size [grid_size + 2 * spline_order + 1]
    spacing = (up_bound - low_bound) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, device=device)
    grid = grid * spacing + low_bound

    # [out_dim, in_dim, G + 2k + 1]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.device = device
        self.grid_range = grid_range
        # Generate (out, in) copies of equally spaced control points on [a, b]
        grid = generate_control_points(
            grid_range[0],
            grid_range[1],
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            device,
        )
        self.register_buffer("grid", grid)

        # Define the univariate B-spline function
        self.univarate_fn = compute_bspline

        # Spline parameters
        self.coef = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim, grid_size + spline_order)
        )

        self._initialization()

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
        # [bsz x in_dim] to [bsz x out_dim x in_dim x (grid_size + spline_order)]
        bases = self.univarate_fn(x, self.grid, self.spline_order, self.device)

        # [bsz x out_dim x in_dim x (grid_size + spline_order)]
        postacts = bases * self.coef[None, ...]

        # [bsz x out_dim x in_dim] to [bsz x out_dim]
        spline = torch.sum(postacts, dim=-1)

        return spline

    def grid_extension(self, x: torch.Tensor, new_grid_size: int):
        """
        Increase granularity of B-spline activation by increasing the
        number of grid points while maintaining the spline shape.
        """

        # Re-generate grid points with extended size (uniform)
        new_grid = generate_control_points(
            self.grid_range[0],
            self.grid_range[1],
            self.in_dim,
            self.out_dim,
            self.spline_order,
            new_grid_size,
            self.device,
        )

        # bsz x out_dim x in_dim x (old_grid_size + spline_order)
        old_bases = self.univarate_fn(x, self.grid, self.spline_order, self.device)

        # bsz x out_dim x in_dim x (new_grid_size + spline_order)
        bases = self.univarate_fn(x, new_grid, self.spline_order, self.device)
        # out_dim x in_dim x bsz x (new_grid_size + spline_order)
        bases = bases.permute(1, 2, 0, 3)

        # bsz x out_dim x in_dim
        postacts = torch.sum(old_bases * self.coef[None, ...], dim=-1)
        # out_dim x in_dim x bsz
        postacts = postacts.permute(1, 2, 0)

        # solve for X in AX = B, A is bases and B is postacts
        new_coefs = torch.linalg.lstsq(
            bases.to(self.device),
            postacts.to(self.device),
            driver="gelsy" if self.device == "cpu" else "gelsd",
        ).solution

        # Set new parameters
        self.grid_size = new_grid_size
        self.grid = new_grid
        self.coef = torch.nn.Parameter(new_coefs, requires_grad=True)



class WeightedResidualLayer(nn.Module):
    """
    Defines the activation function used in the paper,
    phi(x) = w_b SiLU(x) + w_s B_spline(x)
    as a layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual_std: float = 0.1,
    ):
        super(WeightedResidualLayer, self).__init__()
        self.univariate_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_s in paper

        # Residual activation functions
        self.residual_fn = F.silu
        self.residual_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_b in paper

        self._initialization(residual_std)

    def _initialization(self, residual_std):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.normal_(self.residual_weight, mean=0.0, std=residual_std)
        nn.init.ones_(self.univariate_weight)

    def forward(self, x: torch.Tensor, post_acts: torch.Tensor):
        """
        Given the input to a KAN layer and the activation (e.g. spline(x)),
        compute a weighted residual.

        x has shape (bsz, in_dim) and act has shape (bsz, out_dim, in_dim)
        """

        # Broadcast the input along out_dim of post_acts
        res = self.residual_weight * self.residual_fn(x[:, None, :])
        act = self.univariate_weight * post_acts
        return res + act

class KANSymbolic(nn.Module):
    "Defines and stores the Symbolic functions fixed / set for a KAN."

    def __init__(self, in_dim: int, out_dim: int, device: torch.device):
        """
        We have to store a 2D array of univariate functions, one for each
        edge in the KAN layer. 
        """
        super(KANSymbolic, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fns = [[lambda x: x for _ in range(in_dim)] for _ in range(out_dim)]
    
    def forward(self, x: torch.Tensor):
        """
        Run symbolic activations over all inputs in x, where
        x is of shape (batch_size, in_dim). Returns a tensor of shape
        (batch_size, out_dim, in_dim).
        """
        
        acts = []
        # Really inefficient, try tensorizing later.
        for j in range(self.in_dim):
            act_ins = []
            for i in range(self.out_dim):
                o = torch.vmap(self.fns[i][j])(x[:,[j]]).squeeze(dim=-1)
                act_ins.append(o)
            acts.append(torch.stack(act_ins, dim=-1))
        acts = torch.stack(acts, dim=-1)

        return acts

    def set_symbolic(self, in_index: int, out_index: int, fn):
        """
        Set symbolic function at specified edge to new function.
        """
        self.fns[out_index][in_index] = fn 


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

        # Define univariate function (splines in original KAN)
        self.activation_fn = KANActivation(
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            device,
            grid_range,
        )
        
        self.symbolic_fn = KANSymbolic(
            in_dim,
            out_dim,
            device
        )

        self.activation_mask = nn.Parameter(
            torch.ones((out_dim, in_dim), device=device)
        ).requires_grad_(False)
        self.symbolic_mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False)

        # Define the residual connection layer used to compute \phi
        self.residual_layer = WeightedResidualLayer(in_dim, out_dim, residual_std)

        # Cache for regularization
        self.inp = torch.empty(0)
        self.activations = torch.empty(0)

    def cache(self, inp: torch.Tensor, acts: torch.Tensor):
        self.inp = inp
        self.activations = acts

    def set_symbolic(self, in_index: int, out_index: int, fix:bool, fn):
        """
        Set the symbolic mask to be fixed (fix=1) or unfixed. 
        """
        if fix:
            self.symbolic_mask[out_index, in_index] = 1
            self.symbolic_fn.set_symbolic(in_index, out_index, fn)
        else:
            self.symbolic_mask[out_index, in_index] = 0


    def forward(self, x: torch.Tensor):
        """
        Forward pass of KAN. x is expected to be of shape (batch_size, input_size)
        where input_size is the number of input scalars.

        Stores the activations needed for computing the L1 regularization and
        entropy regularization terms.

        Returns the output of the KAN operation.
        """

        spline = self.activation_fn(x)

        # Form the batch of matrices phi(x) of shape [batch_size x out_dim x in_dim]
        phi = self.residual_layer(x, spline)

        # Perform symbolic computations
        sym_phi = self.symbolic_fn(x)
        phi = phi * (self.symbolic_mask == 0) + sym_phi * self.symbolic_mask

        # Mask out pruned edges
        phi = phi * self.activation_mask[None, ...]

        # Cache activations for regularization during training.
        # Also useful for visualizing. Can remove for inference.
        self.cache(x, phi)

        # Really inefficient matmul
        out = torch.sum(phi, dim=-1)

        return out

    def grid_extension(self, x: torch.Tensor, new_grid_size: int):
        """
        Increase granularity of B-spline by increasing the
        number of grid points while maintaining the spline shape.
        """

        self.grid_size = new_grid_size
        self.activation_fn.grid_extension(x, new_grid_size)


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
