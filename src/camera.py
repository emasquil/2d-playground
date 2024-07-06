import matplotlib.pyplot as plt
import numpy as np

from src.scene_generation import INFINITE_DEPTH


def compute_tform_cam2world_2d(viewpoint, center):
    vx, vy = viewpoint
    cx, cy = center

    # Compute the forward vector (from viewpoint to center) and normalize it
    forward = np.array([cx - vx, cy - vy])
    forward = forward / np.linalg.norm(forward)

    # Compute the right vector (perpendicular to forward)
    right = np.array([forward[1], -forward[0]])

    # Rotation matrix from camera to world
    rotation_matrix = np.stack([right, forward], axis=1)

    # Translation vector (camera position in world coordinates)
    translation_vector = np.array([vx, vy])

    # Combine into a 3x3 transformation matrix
    tform_cam2world = np.eye(3)
    tform_cam2world[:2, :2] = rotation_matrix
    tform_cam2world[:2, 2] = translation_vector

    return tform_cam2world.astype("float32")


def take_1d_picture(
    scene, center, viewpoint, focal_length, picture_size, picture_fov, debug=False
):
    # Center of the image
    cx, cy = center
    # Viewpoint coordinates
    vx, vy = viewpoint
    # Pixels of the picture
    projected_pixels = np.zeros((picture_size, 3), dtype=np.float32)
    # Depth map
    depth_map = np.ones((picture_size), dtype=np.float32) * INFINITE_DEPTH
    # Define the image plane points and ray cast
    dx, dy = np.array([cx - vx, cy - vy])
    ux, uy = np.array([dx, dy]) / np.linalg.norm([dx, dy])
    perp_x, perp_y = uy, -ux
    t_values = np.linspace(-picture_fov / 2, picture_fov / 2, picture_size)
    image_plane_points_x = []
    image_plane_points_y = []

    if debug:
        plt.imshow(scene, origin="lower")
        plt.plot(viewpoint[0], viewpoint[1], "rx")
    for j, t in enumerate(t_values):
        x = vx + t * perp_x + focal_length * ux
        y = vy + t * perp_y + focal_length * uy
        image_plane_points_x.append(x)
        image_plane_points_y.append(y)
        # Ray cast
        for n in np.linspace(
            0, round(np.sqrt(scene.shape[0] ** 2 + scene.shape[1] ** 2)), 1000
        ):
            rx, ry = np.array([x - vx, y - vy])
            rx, ry = np.array([rx, ry]) / np.linalg.norm([rx, ry])
            # Find the intersection point
            intersect_x = int(vx + n * rx)
            intersect_y = int(vy + n * ry)
            if debug:
                plt.plot(intersect_x, intersect_y, "y.")
            # Check if the ray is hitting the object
            if (
                0 <= intersect_y < scene.shape[0]
                and 0 <= intersect_x < scene.shape[1]
                and np.sum(scene[intersect_y, intersect_x]) != 0
            ):
                projected_pixels[j] = scene[intersect_y, intersect_x]
                depth_map[j] = n
                break
    # Compute the 3x3 transformation matrix from camera to world coordinates
    tform_cam2world = compute_tform_cam2world_2d(viewpoint, center)
    if debug:
        plt.plot(image_plane_points_x, image_plane_points_y, "b-")
        plt.show()
    return (
        projected_pixels,
        image_plane_points_x,
        image_plane_points_y,
        tform_cam2world,
        depth_map,
    )
