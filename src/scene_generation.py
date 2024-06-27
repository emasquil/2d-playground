import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from typing import Optional, Tuple
from moviepy.editor import ImageSequenceClip
import random

WORLD_SIZE = 400
CENTER = (WORLD_SIZE // 2, WORLD_SIZE // 2)
INFINITE_DEPTH = 400


# Function to generate synthetic 2D scene
def generate_random_scene():
    center = 200, 200
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
    center = 200, 200
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
    center = 200, 200
    # Create a blank canvas
    scene = np.zeros((400, 400, 3), dtype=np.float32)

    # Create a rectangle in the centre with 4 colors one for each quadrant
    scene[100:200, 150:200, :] = [1.0, 0.0, 0.0]
    scene[100:200, 200:250, :] = [0.0, 1.0, 0.0]
    scene[200:300, 150:200, :] = [0.0, 0.0, 1.0]
    scene[200:300, 200:250, :] = [1.0, 1.0, 1.0]

    return scene
