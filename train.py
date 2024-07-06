"""
Main training script for all methods.
"""

import io
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import yaml
from moviepy.editor import ImageSequenceClip
from PIL import Image
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from src import camera, common, nerf, neus, scene_generation

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    method: str = typer.Argument(..., help="Method to train."),
    config: str = typer.Argument(..., help="Path to configuration file."),
    output: str = typer.Argument(..., help="Path to output directory."),
):
    """
    Main training script for all methods.
    """

    # Seed everything
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Torch settings
    torch.autograd.set_detect_anomaly(True)

    # Read configuration file
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # Logging
    writer = SummaryWriter(output)

    # Add scene configs to the config dict
    config["scene"] = {
        "center": scene_generation.CENTER,
        "world_size": scene_generation.WORLD_SIZE,
        "inf_depth": scene_generation.INFINITE_DEPTH,
    }

    # Scene generation
    typer.echo("Generating scene...")
    scene = np.clip(
        scene_generation.generate_deterministic_scene()
        + scene_generation.generate_random_scene()
        - 0.7,
        0,
        1,
    )
    scene = scene_generation.generate_deterministic_scene()
    # If debug let's also get the gt radiance field
    if config["debug"]:
        gt_radiance_field = scene_generation.get_ground_truth_radiance_field(scene)

    # Picture acquisition
    scene_center = config["scene"]["center"]
    theta = np.linspace(
        0,
        2 * np.pi - (2 * np.pi) / config["camera"]["number_of_images"],
        config["camera"]["number_of_images"],
    )
    c_xs = np.cos(theta) * config["camera"]["radius"] + scene_center[0]
    c_ys = np.sin(theta) * config["camera"]["radius"] + scene_center[1]
    images = []
    depth_maps = []
    camera_matrices = []
    plt.imshow(scene, origin="lower")
    arrow_number = 0
    for c_x, c_y in track(
        zip(c_xs, c_ys), description="Taking pictures...", total=len(c_xs)
    ):
        image, _, _, camera_matrix, depth_map = camera.take_1d_picture(
            scene,
            scene_center,
            (c_x, c_y),
            config["camera"]["focal_length"],
            config["camera"]["picture_size"],
            config["camera"]["picture_fov"],
        )
        images.append(image)
        # normalize depth map
        depth_maps.append(depth_map / camera.INFINITE_DEPTH)
        camera_matrices.append(camera_matrix)
        plt.arrow(
            c_x,
            c_y,
            (scene_center[0] - c_x) * 0.1,
            (scene_center[1] - c_y) * 0.1,
            head_width=5,
            head_length=5,
            fc="red",
            ec="red",
        )
        plt.annotate(
            str(arrow_number),
            (c_x, c_y),
            color="red",
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
        )
        arrow_number += 1
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    image = Image.open(buf)
    writer.add_image("Scene", np.moveaxis(np.array(image), -1, 0), 0)
    typer.echo("Scene and camera positions saved.")

    # Preprocess data and separate test set
    images = np.array(images)
    depth_maps = np.array(depth_maps)
    camera_matrices = np.array(camera_matrices)
    test_index = np.random.randint(0, len(images))
    test_image, test_camera_matrix, test_depth_map = (
        images[test_index],
        camera_matrices[test_index],
        depth_maps[test_index],
    )
    images = np.delete(images, test_index, axis=0)
    camera_matrices = np.delete(camera_matrices, test_index, axis=0)
    depth_maps = np.delete(depth_maps, test_index, axis=0)
    # Send everything to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_image = torch.from_numpy(test_image).to(device)
    test_camera_matrix = torch.from_numpy(test_camera_matrix).to(device)
    test_depth_map = torch.from_numpy(test_depth_map).to(device)
    images = torch.from_numpy(images).to(device)
    camera_matrices = torch.from_numpy(camera_matrices).to(device)
    depth_maps = torch.from_numpy(depth_maps).to(device)

    # Precompute all ray bundles for all cameras
    typer.echo("Precomputing ray bundles for all train cameras...")
    rays = [
        common.get_ray_bundle_2d(
            config["camera"]["picture_size"],
            config["camera"]["picture_fov"],
            config["camera"]["focal_length"],
            camera_matrix,
        )
        for camera_matrix in camera_matrices
    ]
    ray_origins, ray_directions = zip(*rays)
    ray_origins = torch.stack(ray_origins, axis=0).view(-1, 2)
    ray_directions = torch.stack(ray_directions, axis=0).view(-1, 2)
    # Precompute all targets for each ray
    targets = []
    targets_depth_map = []
    for i in range(len(images)):
        for j in range(config["camera"]["picture_size"]):
            targets.append(images[i][j])
            targets_depth_map.append(depth_maps[i][j])
    targets = torch.stack(targets)
    targets_depth_map = torch.stack(targets_depth_map)

    # Model initialization
    if method == "neus":
        raise NotImplemented("Method not implemented yet.")
    elif method == "nerf":
        model = nerf.VeryTinyNerfModel2D()
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["training"]["lr"])
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, config["training"]["num_iters"]
        # )

    typer.echo("Starting training...")
    for i in track(range(config["training"]["num_iters"]), description="Training..."):
        # for i in range(config["training"]["num_iters"]):
        if config["training"]["random_batches"]:
            # Create a batch of rays and targets from multiple images
            batch_indices = np.random.choice(
                len(ray_origins), config["training"]["batch_size"]
            )
            ray_o = ray_origins[batch_indices]
            ray_d = ray_directions[batch_indices]
            target_rgb = targets[batch_indices]
            target_depth = targets_depth_map[batch_indices]
        else:
            # Consider a single image as a target
            target_image_index = np.random.randint(len(images))
            ray_o = ray_origins[
                target_image_index
                * config["camera"]["picture_size"] : (target_image_index + 1)
                * config["camera"]["picture_size"]
            ]
            ray_d = ray_directions[
                target_image_index
                * config["camera"]["picture_size"] : (target_image_index + 1)
                * config["camera"]["picture_size"]
            ]
            target_rgb = images[target_image_index]
            target_depth = depth_maps[target_image_index]

        # Forward pass
        query_points, depth_values = common.compute_query_points_from_rays_2d(
            ray_o,
            ray_d,
            config["training"]["near_thresh"],
            config["training"]["far_thresh"],
            config["training"]["depth_samples_per_ray"],
        )
        flattened_query_points = query_points.view(-1, 2)  # W x 2
        # Normalize coords between 0 and 1
        flattened_query_points = (
            flattened_query_points / config["scene"]["world_size"]
        ) * 2 - 1  # W x 2
        encoded_query_points = common.positional_encoding(
            flattened_query_points
        )  # W x 64
        batches = common.get_minibatches(
            encoded_query_points, chunksize=config["training"]["chunksize"]
        )  # List of W x 64
        predictions = []
        for batch in batches:
            if method == "neus":
                raise NotImplementedError("Method not implemented yet.")
            elif method == "nerf":
                predictions.append(model(batch))

        # Backward pass
        if method == "neus":
            raise NotImplementedError("Method not implemented yet.")
        elif method == "nerf":
            radiance_field_flattened = torch.cat(predictions, dim=0)  # W x 4
            unflattened_shape = list(query_points.shape[:-1]) + [4]  # H x W x 4
            radiance_field = radiance_field_flattened.view(
                unflattened_shape
            )  # H x W x 4
            rgb_predicted, depth_predicted, _ = nerf.render_volume_density_2d(
                radiance_field,
                ray_o,
                depth_values,
            )  # W x 3, W
            rgb_loss = torch.nn.functional.mse_loss(rgb_predicted, target_rgb)
            # Normalize predicted depth
            depth_predicted /= camera.INFINITE_DEPTH
            # If monocular cue is enabled, add depth loss
            if config["training"]["monocular_cue"]:
                depth_loss = torch.nn.functional.mse_loss(depth_predicted, target_depth)
                loss = rgb_loss + depth_loss * 0.1
            else:
                loss = rgb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Scheduler step
            # scheduler.step()
            # Log to tensorboard
            writer.add_scalar("Loss/train", loss.item(), i)

        # Tensorboard eval
        if i % config["training"]["display_every"] == 0:
            # Render the held-out view
            with torch.no_grad():
                test_ray_o, test_ray_d = common.get_ray_bundle_2d(
                    config["camera"]["picture_size"],
                    config["camera"]["picture_fov"],
                    config["camera"]["focal_length"],
                    test_camera_matrix,
                )
                test_query_points, test_depth_values = (
                    common.compute_query_points_from_rays_2d(
                        test_ray_o,
                        test_ray_d,
                        config["training"]["near_thresh"],
                        config["training"]["far_thresh"],
                        config["training"]["depth_samples_per_ray"],
                    )
                )
                test_flattened_query_points = test_query_points.view(-1, 2)
                test_flattened_query_points_normalized = (
                    test_flattened_query_points / config["scene"]["world_size"]
                ) * 2 - 1
                test_encoded_query_points = common.positional_encoding(
                    test_flattened_query_points_normalized
                )
                test_batches = common.get_minibatches(
                    test_encoded_query_points, chunksize=config["training"]["chunksize"]
                )

                if method == "neus":
                    raise NotImplementedError("Method not implemented yet.")
                elif method == "nerf":
                    if config["debug"]:
                        # Get the gt radiance field for the query points
                        test_radiance_field_flattened = torch.zeros(
                            test_flattened_query_points.shape[0], 4
                        )
                        for idx in range(test_flattened_query_points.shape[0]):
                            nearest_x = round(
                                test_flattened_query_points[idx, 0].item()
                            )
                            nearest_y = round(
                                test_flattened_query_points[idx, 1].item()
                            )
                            if nearest_x < 0 or nearest_x >= scene.shape[1]:
                                continue
                            if nearest_y < 0 or nearest_y >= scene.shape[0]:
                                continue
                            test_radiance_field_flattened[idx] = torch.tensor(
                                gt_radiance_field[nearest_x, nearest_y, :]
                            )

                    else:
                        # Get the radiance field from the model
                        test_predictions = []
                        for batch in test_batches:
                            test_predictions.append(model(batch))
                        test_radiance_field_flattened = torch.cat(
                            test_predictions, dim=0
                        )

                    test_unflattened_shape = list(test_query_points.shape[:-1]) + [4]
                    test_radiance_field = test_radiance_field_flattened.view(
                        test_unflattened_shape
                    )

                    test_rgb_predicted, test_depth_predicted, _ = (
                        nerf.render_volume_density_2d(
                            test_radiance_field,
                            test_ray_o,
                            test_depth_values,
                            debug=config["debug"],
                        )
                    )
                    test_rgb_loss = torch.nn.functional.mse_loss(
                        test_rgb_predicted, test_image
                    )
                    # Normalize predicted depth
                    test_depth_predicted /= camera.INFINITE_DEPTH
                    # If monocular cue is enabled, add depth loss
                    if config["training"]["monocular_cue"]:
                        # Normalize depth values
                        test_depth_loss = torch.nn.functional.mse_loss(
                            test_depth_predicted, test_depth_map
                        )
                        test_loss = test_rgb_loss + test_depth_loss * 0.1
                    else:
                        test_loss = test_rgb_loss

                    # Sample density and rgb in all world points
                    # Generate 2D coordinates for all points in the world
                    world_points = torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                0,
                                config["scene"]["world_size"] - 1,
                                config["scene"]["world_size"],
                            ),
                            torch.linspace(
                                0,
                                config["scene"]["world_size"] - 1,
                                config["scene"]["world_size"],
                            ),
                        ),
                        dim=-1,
                    ).to(
                        device
                    )  # H x W x 2
                    # Normalize world points
                    world_points = (
                        world_points / config["scene"]["world_size"]
                    ) * 2 - 1
                    # Flatten world points
                    world_points = world_points.view(-1, 2)
                    # Encode world points
                    world_points = common.positional_encoding(world_points)
                    # Split world points into batches
                    world_batches = common.get_minibatches(
                        world_points, chunksize=config["training"]["chunksize"]
                    )
                    # Predict density and rgb for all world points
                    world_predictions = []
                    for batch in world_batches:
                        world_predictions.append(model(batch))
                    world_radiance_field_flattened = torch.cat(world_predictions, dim=0)
                    world_unflattened_shape = [
                        scene_generation.WORLD_SIZE,
                        scene_generation.WORLD_SIZE,
                    ] + [4]
                    world_radiance_field = world_radiance_field_flattened.view(
                        world_unflattened_shape
                    )
                    if config["debug"]:
                        # If we are debugging let's use the gt radiance field
                        world_radiance_field = torch.tensor(gt_radiance_field)
                    # World radiance field contain values as radiancefield[i,j] = f(i,j)
                    # We want to plot this with i being the horizontal coordinate and j the vertical coordinate
                    # Thus we need to transpose the radiance field
                    world_radiance_field = world_radiance_field.permute(1, 0, 2)

                    if config["debug"]:
                        # Don't apply any activation to the maps
                        density = world_radiance_field[..., 3]
                        rgb = world_radiance_field[..., :3]
                    else:
                        # Apply an activation to the maps
                        density = torch.nn.functional.relu(world_radiance_field[..., 3])
                        rgb = torch.sigmoid(world_radiance_field[..., :3])

                    density = np.expand_dims(
                        density.detach().cpu().numpy(),
                        axis=0,
                    )
                    rgb = np.moveaxis(
                        rgb.detach().cpu().numpy(),
                        -1,
                        0,
                    )

                    # Log density and rgb in tensorboard
                    # We need to flip the images because tensorboard expects the origin to be at the top left corner
                    # and we are using the bottom left corner
                    writer.add_image(
                        "Density map", np.flip(1 - np.exp(-density), axis=1), i
                    )
                    writer.add_image("RGB map", np.flip(rgb, axis=1), i)

                    # Log test loss in tensorboard
                    writer.add_scalar("Loss/test", test_loss.item(), i)
                    # Log test image and rendered image in tensorboard
                    test_image_log = test_image.detach().cpu().numpy()  # W x 3
                    test_image_log = np.tile(test_image_log, (200, 1, 1))  # 200 x W x 3
                    test_image_log = np.moveaxis(test_image_log, -1, 0)  # 3 x 200 x W
                    test_rgb_predicted = (
                        test_rgb_predicted.detach().cpu().numpy()
                    )  # W x 3
                    test_rgb_predicted = np.tile(
                        test_rgb_predicted, (200, 1, 1)
                    )  # 200 x W x 3
                    test_rgb_predicted = np.moveaxis(
                        test_rgb_predicted, -1, 0
                    )  # 3 x 200 x W
                    writer.add_image("Test Image", test_image_log, i)
                    writer.add_image("Rendered Image", test_rgb_predicted, i)

                    # Log depth maps
                    test_depth_map_log = test_depth_map.detach().cpu().numpy()  # W
                    test_depth_map_log = np.tile(
                        test_depth_map_log, (1, 200, 1)
                    )  # 1 x 200 x W
                    test_depth_map_predicted = (
                        test_depth_predicted.detach().cpu().numpy()
                    )  # W
                    test_depth_map_predicted = np.tile(
                        test_depth_map_predicted, (1, 200, 1)
                    )  # 1 x 200 x W
                    writer.add_image("Test Depth Map", test_depth_map_log, i)
                    writer.add_image("Rendered Depth Map", test_depth_map_predicted, i)

    # After training is completed, save the model
    torch.save(model.state_dict(), f"{output}/model.pth")
    typer.echo("Training completed, model saved.")

    # Generate a circular video around the scene
    typer.echo("Generating circular video...")
    camera.generate_videos(
        scene,
        list(zip(c_xs, c_ys)),
        model,
        nerf.nerf_2d,
        output,
        "circular",
        config,
    )

    # Generate a zoom out video from the scene
    starting_point = (150, 80)
    # Compute direction between starting point and center
    video_direction = np.array(scene_center) - np.array(starting_point)
    # Normalize the direction
    direction = video_direction / np.linalg.norm(video_direction)
    depths = np.linspace(0, 200, 50)
    # Generate 50 view points from starting point away from the center
    video_viewpoints = [starting_point - depth * direction for depth in depths]
    typer.echo("Generating zoom out video...")
    camera.generate_videos(
        scene,
        video_viewpoints,
        model,
        nerf.nerf_2d,
        output,
        "zoom_out",
        config,
    )

    # Generate a zoom in video from the scene
    starting_point = (c_xs[39], c_ys[39])
    # Compute direction between center and starting point
    video_direction = np.array(starting_point) - np.array(scene_center)
    # Normalize the direction
    direction = video_direction / np.linalg.norm(video_direction)
    depths = np.linspace(0, 100, 50)
    # Generate 50 view points from starting point away from the center
    video_viewpoints = [starting_point - depth * direction for depth in depths]
    typer.echo("Generating zoom in video...")
    camera.generate_videos(
        scene,
        video_viewpoints,
        model,
        nerf.nerf_2d,
        output,
        "zoom_in",
        config,
    )


if __name__ == "__main__":
    app()
