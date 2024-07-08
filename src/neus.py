from typing import Tuple

import numpy as np
import torch

from .common import (
    compute_query_points_from_rays_2d,
    cumprod_exclusive,
    get_minibatches,
    get_ray_bundle_2d,
    positional_encoding,
)
from .scene_generation import CENTER, WORLD_SIZE


def Phi_s(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(s * x)


class DistanceModel(torch.nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(DistanceModel, self).__init__()
        self.layer1 = torch.nn.Linear(2 + 2 * 2 * num_encoding_functions, filter_size)
        # Layer 2
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3: output 1 value for distance
        self.layer3 = torch.nn.Linear(filter_size, 1)
        # Activation function
        self.softplus = torch.nn.functional.softplus

    def forward(self, x):
        x = self.softplus(self.layer1(x))
        x = self.softplus(self.layer2(x))
        x = self.layer3(x)
        return self.softplus(x)


class ColorModel(torch.nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(ColorModel, self).__init__()
        # Input layer: input dimension is 2 for 2D coordinates but they are encoded, we duplicate the input because we also have the gradient
        self.layer1 = torch.nn.Linear(2 * (2 + 2 * 2 * num_encoding_functions), filter_size)
        # Layer 2
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3: output 3 values for RGB
        self.layer3 = torch.nn.Linear(filter_size, 3)
        # Activation function
        self.softplus = torch.nn.functional.softplus

    def forward(self, x):
        x = self.softplus(self.layer1(x))
        x = self.softplus(self.layer2(x))
        x = self.layer3(x)
        return torch.nn.functional.sigmoid(x)


class SModel(torch.nn.Module):
    def __init__(self, variance_init=0.5, speed_factor=10.0):
        super(SModel, self).__init__()
        # self.ln_s = torch.nn.Parameter(
        #     data=torch.Tensor([-np.log(variance_init) / speed_factor]),
        #     requires_grad=True,
        # )
        # self.speed_factor = speed_factor
        # TODO: remove
        self.s = torch.nn.Parameter(
            data=torch.Tensor([variance_init]), requires_grad=True
        )

    def forward(self):
        # return torch.exp(self.ln_s * self.speed_factor)
        return torch.exp(self.s * 10.0)


# class NeusModel(torch.nn.Module):
#     def __init__(self, variance_init=0.05, speed_factor=1.0):
#         # TODO: understand the following line
#         super(NeusModel, self).__init__()

#         # learnable s parameter
#         self.ln_s = torch.nn.Parameter(
#             data=torch.Tensor(
#                 [-np.log(variance_init) / speed_factor], requires_grad=True
#             )
#         )
#         self.speed_factor = speed_factor

#         # Distance network
#         self.distance_network = DistanceModel()

#         # Color network
#         self.color_network = ColorModel()

#     def forward_s(self):
#         # TODO: whats speed factor
#         return torch.exp(self.ln_s * self.speed_factor)

#     def forward_radiance(self, x: torch.Tensor):
#         sdf, grads = self.distance_network.forward_with_grads(x)
#         radiance = self.color_network.forward(x)


def render_volume_density_2d(
    sdf: torch.Tensor,
    s: torch.Tensor,
    color_field: torch.Tensor,
    depth_values: torch.Tensor,
    debug=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, d = sdf.shape  # b batch size, d 1d picture size
    sigmoid_sdf = Phi_s(s, sdf)  # b, d
    alpha = torch.maximum(
        (sigmoid_sdf[:-1] - sigmoid_sdf[1:]) / sigmoid_sdf[:-1]
        + 1e-10,  # why do we need this small constant?
        torch.zeros_like(sigmoid_sdf[1:]),
    )  # b-1, d, 1
    # add a last 0 as the last alpha which is the one obtained by the difference between
    # the last point and an inifinite distance point #TODO: validate this assumption makes sense,
    # I did the same as with last nerf point
    alpha = torch.cat((alpha, torch.zeros([1, d])), dim=0)  # b, d
    weights = alpha * cumprod_exclusive(1.0 - alpha)  # b, d
    rgb_map = (weights[..., None] * color_field).sum(dim=1)  # b, 3
    depth_map = (weights * depth_values).sum(dim=-1)  # b
    acc_map = weights.sum(dim=-1)  # b
    return rgb_map, depth_map, acc_map
