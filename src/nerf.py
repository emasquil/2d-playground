from typing import Tuple

import torch

from .common import (
    compute_query_points_from_rays_2d,
    cumprod_exclusive,
    get_minibatches,
    get_ray_bundle_2d,
    positional_encoding,
)
from .scene_generation import CENTER, WORLD_SIZE


class VeryTinyNerfModel2D(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers for 2D inputs."""

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel2D, self).__init__()
        # Input layer: input dimension is 2 for 2D coordinates
        self.layer1 = torch.nn.Linear(2 + 2 * 2 * num_encoding_functions, filter_size)
        # Layer 2
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3: output 4 dimensions (RGB + density)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Activation function
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def render_volume_density_2d(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor,
    debug=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
        radiance_field (torch.Tensor): A "field" where, at each query location (X, Y),
          we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
          the paper) (shape: :math:`(picture_size, num_samples, 4)`).
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(picture_size, 2)`).
        depth_values (torch.Tensor): Sampled depth values along each ray
          (shape: :math:`(num_samples)`).

    Returns:
        rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(picture_size, 3)`).
        depth_map (torch.Tensor): Rendered depth image (shape: :math:`(picture_size)`).
        acc_map (torch.Tensor): Accumulated transmittance map (shape: :math:`(picture_size)`).
    """
    sigma_a = radiance_field[..., 3]
    rgb = radiance_field[..., :3]
    # If we are debugging the radiance field, we do not apply any activation since they are already ground truth values
    if not debug:
        # If we are not in debug mode, we need to apply an activation to the maps
        sigma_a = torch.nn.functional.softplus(radiance_field[..., 3])
        rgb = torch.sigmoid(radiance_field[..., :3])
    # Large constant for the distance at the last sample point
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    # Compute distances between adjacent depth values
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    # Compute alpha values
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    # Compute weights
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    # Compute rendered RGB map
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    # Compute rendered depth map
    depth_map = (weights * depth_values).sum(dim=-1)
    # Compute accumulated transmittance map
    acc_map = weights.sum(-1)
    return rgb_map, depth_map, acc_map


def nerf_2d(
    camera_matrix,
    picture_size,
    picture_fov,
    focal_length,
    near_thresh,
    far_thresh,
    depth_samples_per_ray,
    nerf_model,
    chunksize,
    nerf_radius,
):
    """
    Helper function that computes the rgb, depth, and acc map for a given pose (camera matrix).

    Args:
    """

    ray_origins, ray_directions = get_ray_bundle_2d(
        picture_size, picture_fov, focal_length, camera_matrix
    )
    query_points, depth_values = compute_query_points_from_rays_2d(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )
    flattened_query_points = query_points.view(-1, 2)
    # Normalize coords between 0 and 1
    flattened_query_points_normalized = (flattened_query_points / WORLD_SIZE) * 2 - 1
    encoded_query_points = positional_encoding(flattened_query_points_normalized)
    batches = get_minibatches(encoded_query_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(nerf_model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)
    unflatten_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = radiance_field_flattened.view(unflatten_shape)
    # TODO: Check if the following makes sense or not, I'm trying to avoid evaluating the model in areas outside the nerf_radius
    # # If ray origin is outside the nerf radius to the following
    # if torch.norm(ray_origins[0] - torch.tensor(CENTER), dim=-1) > nerf_radius:
    #     # Make the radiance field == 0 for all query points outside the nerf_radius
    #     distance = torch.norm(flattened_query_points - torch.tensor(CENTER), dim=-1)
    #     distance = distance.view(query_points.shape[:-1])
    #     radiance_field[distance > nerf_radius] = 0
    # # # Log number of points where distance > nerf_radius
    # # num_points_outside_radius = (distance > nerf_radius).sum()
    # # print(f"Number of points outside radius: {num_points_outside_radius}")
    return render_volume_density_2d(radiance_field, ray_origins, depth_values)
