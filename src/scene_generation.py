import typing as T

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

WORLD_SIZE = 400
CENTER = (WORLD_SIZE // 2, WORLD_SIZE // 2)
INFINITE_DEPTH = 400


def generate_all_green_scene():
    scene = np.zeros((400, 400, 3), dtype=np.float32)

    scene[100:300, 150:250, :] = np.array([0.0, 1.0, 0.0])
    return scene


# Function to generate synthetic 2D scene
def generate_random_scene():
    # Create a blank canvas
    scene = np.zeros((400, 400, 3), dtype=np.float32)

    # Draw a random color rectangle
    scene[100:300, 150:250, :] = (
        np.random.randn(200 * 100 * 3).reshape(200, 100, 3) + 0.7
    )

    # Apply a gaussian filter
    scene = gaussian_filter(scene, sigma=3)
    scene = np.clip(scene, 0, 1.0).astype(np.float32)

    # Make everything black outside of the rectangle
    scene[0:100, :, :] = 0
    scene[301:400, :, :] = 0
    scene[:, 0:150, :] = 0
    scene[:, 251:400, :] = 0
    return scene


def generate_gradient_scene():
    # Create a blank canvas
    scene = np.zeros((400, 400, 3), dtype=np.float32)

    # Define the gradient values for each channel
    num_rows = 300 - 100  # Height of the rectangle
    num_cols = 250 - 150  # Width of the rectangle

    r_gradient = np.linspace(
        1.0, 0.5, num_cols
    )  # Gradient from 1.0 to 0.5 across the width
    g_gradient = np.linspace(
        1.0, 0.7, num_cols
    )  # Gradient from 1.0 to 0.7 across the width
    b_gradient = np.linspace(
        1.0, 0.0, num_cols
    )  # Gradient from 1.0 to 0.0 across the width

    # Stack the gradients to match the shape (width, 3)
    gradient = np.stack((r_gradient, g_gradient, b_gradient), axis=1).astype(np.float32)

    # Repeat the gradient across the height of the rectangle
    gradient = np.tile(gradient, (num_rows, 1, 1))

    # Assign the gradient to the specified slice in the scene array
    scene[100:300, 150:250, :] = gradient

    return scene


def generate_deterministic_scene():
    # Create a blank canvas
    scene = np.zeros((400, 400, 3), dtype=np.float32)

    # Create a rectangle in the centre with 4 colors one for each quadrant
    scene[100:200, 150:200, :] = [1.0, 0.0, 0.0]
    scene[100:200, 200:250, :] = [0.0, 1.0, 0.0]
    scene[200:300, 150:200, :] = [0.0, 0.0, 1.0]
    scene[200:300, 200:250, :] = [1.0, 1.0, 1.0]

    return scene


def get_ground_truth_radiance_field(scene: np.ndarray) -> np.ndarray:
    """Helper function to compute the ground truth density and RGB maps from a scene, to help with debugging.

    Args:
        scene (np.ndarray): The input scene

    Returns:
        np.ndarray: The radiance field containing the rgb as the first 3 channels and the density as the last channel ([H, W, 4])
    """

    # Density should be 0 where the scene is black, and 1 elsewhere
    # sigma[x, y], where x is the horizontal coordinate and y is the vertical coordinate, thus to compute sigma[x,y] we need to check scene[y, x]
    density_map = np.zeros_like(scene[..., 0])
    density_map[scene.sum(axis=-1) > 0] = 1.0
    density_map = density_map.transpose(1, 0)
    # RGB values are the scene values, rgb[x, y, c], where x is the horizontal coordinate, y is the vertical coordinate, and c is the color channel
    # rgb[x, y, c] = scene[y, x, c]
    rgb_map = scene.transpose(1, 0, 2)
    radiance_field = np.concatenate([rgb_map, density_map[..., None]], axis=-1)
    return radiance_field


def get_ground_truth_sdf(
    scene: np.ndarray, epsilon: float = 2
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ground truth signed distance function (SDF) for the scene and the rgb map.

    Args:
    scene (np.ndarray): The input scene of shape (H, W, 3)
    epsilon (float): Threshold for considering a value as 0 (boundary)

    Returns:
    np.ndarray: The SDF map of shape (w, H)
    np.ndarray: The RGB map of shape (W, H, 3)
    """

    H, W, _ = scene.shape

    # Create a binary mask of the scene
    mask = (scene.sum(axis=-1) > 0).astype(np.uint8)

    # Compute the distance transform for inside and outside
    distance_inside = distance_transform_edt(mask)
    distance_outside = distance_transform_edt(1 - mask)

    # Combine the inside and outside distances to create the SDF
    sdf = distance_outside - distance_inside

    # Set values close to 0 to exactly 0
    sdf[np.abs(sdf) < epsilon] = 0  # H, W

    # RGB values are the scene values, rgb[x, y, c], where x is the horizontal coordinate, y is the vertical coordinate, and c is the color channel
    # rgb[x, y, c] = scene[y, x, c]
    rgb_map = scene.transpose(1, 0, 2)  # W, H, 3

    return sdf.T, rgb_map
