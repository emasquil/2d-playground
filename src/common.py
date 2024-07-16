from typing import Optional, Tuple

import numpy as np
import torch


def get_ray_bundle_2d(
    picture_size: int,
    picture_fov: int,
    focal_length: float,
    tform_cam2world: np.ndarray,
):
    # Generate pixel coordinates for the 1D picture
    t_values = torch.linspace(-picture_fov / 2, picture_fov / 2, picture_size)

    # Normalize pixel coordinates
    directions = torch.stack(
        [t_values / focal_length, torch.ones_like(t_values)], dim=-1
    )

    # Rotate and translate to world coordinates
    rotation_matrix = tform_cam2world[:2, :2]
    translation_vector = tform_cam2world[:2, 2]

    ray_directions = torch.matmul(directions.cpu(), rotation_matrix.T.cpu())
    ray_origins = torch.tile(translation_vector.cpu(), (picture_size, 1))

    return ray_origins, ray_directions


def compute_query_points_from_rays_2d(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute query 2D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 2D points are to be sampled.

    Args:
      ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_ray_bundle()` method (shape: :math:`(picture_size, 2)`).
      ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_ray_bundle()` method (shape: :math:`(picture_size, 2)`).
      near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
      far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
      num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
      randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray in the "bundle".

    Returns:
      query_points (torch.Tensor): Query points along each ray
        (shape: :math:`(picture_size, num_samples, 2)`).
      depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(num_samples)`).
    """
    # Shape: (num_samples)
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        # ray_origins: (picture_size, 2)
        # noise_shape = (picture_size, num_samples)
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        # depth_values: (num_samples)
        depth_values = (
            depth_values
            + torch.rand(noise_shape).to(ray_origins)
            * (far_thresh - near_thresh)
            / num_samples
        )

    # (picture_size, num_samples, 2) = (picture_size, 1, 2) + (picture_size, 1, 2) * (num_samples, 1)
    # query_points: (picture_size, num_samples, 2)
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    return query_points, depth_values


def cumprod_exclusive(tensor):
    r"""
    Compute exclusive cumulative product of the input tensor along the last dimension.
    """
    # Compute the cumulative product along the last dimension and prepend 1 along the last dimension
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.cat([torch.ones_like(cumprod[..., :1]), cumprod[..., :-1]], dim=-1)
    return cumprod


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
      tensor (torch.Tensor): Input tensor to be positionally encoded.
      num_encoding_functions (optional, int): Number of encoding functions used to
          compute a positional encoding (default: 6).
      include_input (optional, bool): Whaether or not to include the input in the
          computed positional encoding (default: True).
      log_sampling (optional, bool): Sample logarithmically in frequency space, as
          opposed to linearly (default: True).

    Returns:
      (torch.Tensor): Positional encoding of the input tensor.
    """
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))
    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
