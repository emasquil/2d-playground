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
from PIL import Image
from rich.progress import track
from schedulefree import AdamWScheduleFree
from torch.utils.tensorboard import SummaryWriter

from src import camera, common, nerf, neus, scene_generation

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    method: str = typer.Argument(..., help="Method to train."),
    config: str = typer.Argument(..., help="Path to configuration file."),
    output: str = typer.Argument(..., help="Path to output directory."),
    gpu_number: int = typer.Option(0, help="GPU number to use."),
):
    """
    Main training script for all methods.
    """

    # Seed everything
    seed = 99
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

    # Save config in tensorboard
    writer.add_text("Config", yaml.dump(config), 0)

    # Scene generation
    typer.echo("Generating scene...")
    # scene = np.clip(
    #     scene_generation.generate_deterministic_scene()
    #     + scene_generation.generate_random_scene()
    #     - 0.7,
    #     0,
    #     1,
    # )
    scene = scene_generation.generate_deterministic_scene()
    # scene = scene_generation.generate_random_scene()
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
    if torch.cuda.is_available():
        device = torch.device(gpu_number)
    else:
        device = torch.device("cpu")
    typer.echo(f"Training on {device}.")
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
    ray_origins = torch.stack(ray_origins, axis=0).view(-1, 2).to(device)
    ray_directions = torch.stack(ray_directions, axis=0).view(-1, 2).to(device)
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
        distance_model = neus.DistanceModel()
        s_model = neus.SModel()
        color_model = neus.ColorModel()
        # Send all models to device and set up optimizer
        distance_model.to(device)
        s_model.to(device)
        color_model.to(device)
        optimizer = AdamWScheduleFree(
            list(distance_model.parameters())
            + list(s_model.parameters())
            + list(color_model.parameters()),
            lr=float(config["training"]["lr"]),
            warmup_steps=config["training"]["warmup_steps"],
        )

    elif method == "nerf":
        model = nerf.VeryTinyNerfModel2D()
        model.to(device)
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=float(config["training"]["lr"]),
            warmup_steps=config["training"]["warmup_steps"],
        )

    typer.echo("Starting training...")
    for i in track(range(config["training"]["num_iters"]), description="Training..."):
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
        if method == "neus":
            # We need the gradients for the eikonal loss
            query_points.requires_grad_(True)
        # W width of the image, D depth samples per ray
        flattened_query_points = query_points.view(-1, 2)  # (W * D) x 2
        # Normalize coords between 0 and 1
        flattened_query_points = (
            flattened_query_points / config["scene"]["world_size"]
        ) * 2 - 1  # (W * D) x 2
        batches = common.get_minibatches(
            flattened_query_points, chunksize=config["training"]["chunksize"]
        )  # List of W x 2
        predictions = []
        predicted_sdfs = []
        predicted_colors = []
        gradients_sdfs = []
        for batch in batches:
            if method == "neus":
                predicted_sdfs.append(distance_model(batch))
                gradients_sdfs.append(
                    torch.autograd.grad(
                        outputs=predicted_sdfs[-1],
                        inputs=batch,
                        grad_outputs=torch.ones_like(predicted_sdfs[-1]),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                )
                predicted_colors.append(color_model(batch))
            elif method == "nerf":
                predictions.append(model(batch))

        # Backward pass
        if method == "neus":
            sdf_flattened = torch.cat(predicted_sdfs, dim=0)  # (W * D) x 1
            color_flattened = torch.cat(predicted_colors, dim=0)  # (W * D) x 3
            unflattened_distance_shape = list(query_points.shape[:-1])  # W x D
            sdf = sdf_flattened.view(unflattened_distance_shape)  # W x D
            unflattened_color_shape = list(query_points.shape[:-1]) + [3]  # W x D x 3
            color = color_flattened.view(unflattened_color_shape)  # W x D x 3
            gradients_flattened = torch.cat(gradients_sdfs, dim=0)  # (W * D) x f
            unflattened_gradients_shape = list(query_points.shape[:-1]) + [
                gradients_flattened.shape[-1]
            ]  # W x D x f
            gradients = gradients_flattened.view(
                unflattened_gradients_shape
            )  # W x D x f
            s = s_model()  # 1, just a scalar
            rgb_predicted, depth_predicted, _ = neus.render_volume_density_2d(
                sdf, s, color, depth_values
            )
            # Compute rgb and monocular loss first
            rgb_loss = torch.nn.functional.l1_loss(
                rgb_predicted, target_rgb, reduction="sum"
            )
            # Normalize predicted depth
            depth_predicted /= camera.INFINITE_DEPTH
            # If monocular cue is enabled, add depth loss
            if config["training"]["monocular_cue"]:
                depth_loss = torch.nn.functional.l1_loss(depth_predicted, target_depth)
                loss = rgb_loss + depth_loss * 0.1
            else:
                loss = rgb_loss
            # Compute eikonal loss
            eikonal_loss = torch.nn.functional.mse_loss(
                torch.norm(gradients, dim=-1), torch.ones_like(gradients[..., 0])
            )
            loss += eikonal_loss * config["training"]["eikonal_weight"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Log to tensorboard
            writer.add_scalar("Loss/train", loss.item(), i)
            writer.add_scalar("Variance/train", 1 / s.item(), i)

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
            # Log to tensorboard
            writer.add_scalar("Loss/train", loss.item(), i)

        # Tensorboard eval
        if i % config["training"]["display_every"] == 0:
            # Render the held-out view
            test_ray_o, test_ray_d = common.get_ray_bundle_2d(
                config["camera"]["picture_size"],
                config["camera"]["picture_fov"],
                config["camera"]["focal_length"],
                test_camera_matrix,
            )
            test_query_points, test_depth_values = (
                common.compute_query_points_from_rays_2d(
                    test_ray_o.to(device),
                    test_ray_d.to(device),
                    config["training"]["near_thresh"],
                    config["training"]["far_thresh"],
                    config["training"]["depth_samples_per_ray"],
                )
            )
            if method == "neus":
                test_query_points.requires_grad_(True)

            test_flattened_query_points = test_query_points.view(-1, 2)
            test_flattened_query_points_normalized = (
                test_flattened_query_points / config["scene"]["world_size"]
            ) * 2 - 1
            test_batches = common.get_minibatches(
                test_flattened_query_points_normalized,
                chunksize=config["training"]["chunksize"],
            )

            if method == "neus":
                # Save models
                torch.save(distance_model.state_dict(), f"{output}/distance_model.pth")
                torch.save(s_model.state_dict(), f"{output}/s_model.pth")
                torch.save(color_model.state_dict(), f"{output}/color_model.pth")
                # Get the sdf and rgb map from the model
                test_sdf_predictions = []
                test_color_predictions = []
                test_gradients = []
                for batch in test_batches:
                    test_sdf_predictions.append(distance_model(batch))
                    test_gradients.append(
                        torch.autograd.grad(
                            outputs=test_sdf_predictions[-1],
                            inputs=batch,
                            grad_outputs=torch.ones_like(test_sdf_predictions[-1]),
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                    )
                    test_color_predictions.append(color_model(batch))
                test_sdf_flattened = torch.cat(test_sdf_predictions, dim=0)
                test_color_flattened = torch.cat(test_color_predictions, dim=0)
                test_gradients_flattened = torch.cat(test_gradients, dim=0)

                test_unflattened_distance_shape = list(test_query_points.shape[:-1])
                test_sdf = test_sdf_flattened.view(test_unflattened_distance_shape)
                test_unflattened_color_shape = list(test_query_points.shape[:-1]) + [3]
                test_color = test_color_flattened.view(test_unflattened_color_shape)
                test_unflattened_gradients_shape = list(
                    test_query_points.shape[:-1]
                ) + [test_gradients_flattened.shape[-1]]
                test_gradients = test_gradients_flattened.view(
                    test_unflattened_gradients_shape
                )
                test_s = s_model()
                test_rgb_predicted, test_depth_predicted, _ = (
                    neus.render_volume_density_2d(
                        test_sdf, test_s, test_color, test_depth_values
                    )
                )
                test_rgb_loss = torch.nn.functional.mse_loss(
                    test_rgb_predicted, test_image
                )
                # Normalize predicted depth
                test_depth_predicted /= camera.INFINITE_DEPTH
                # If monocular cue is enabled, add depth loss
                if config["training"]["monocular_cue"]:
                    test_depth_loss = torch.nn.functional.mse_loss(
                        test_depth_predicted, test_depth_map
                    )
                    test_loss = test_rgb_loss + test_depth_loss * 0.1
                else:
                    test_loss = test_rgb_loss
                # Compute eikonal loss
                test_eikonal_loss = torch.nn.functional.mse_loss(
                    torch.norm(test_gradients, dim=-1),
                    torch.ones_like(test_gradients[..., 0]),
                )
                test_loss += test_eikonal_loss * config["training"]["eikonal_weight"]
                # Log to tensorboard
                writer.add_scalar("Loss/test", test_loss.item(), i)
                # Log test image and rendered image in tensorboard
                test_image_log = test_image.detach().cpu().numpy()  # W x 3
                test_image_log = np.tile(test_image_log, (200, 1, 1))  # 200 x W x 3
                test_image_log = np.moveaxis(test_image_log, -1, 0)  # 3 x 200 x W
                test_rgb_predicted = test_rgb_predicted.detach().cpu().numpy()  # W x 3
                test_rgb_predicted = np.tile(
                    test_rgb_predicted, (200, 1, 1)
                )  # 200 x W x 3
                test_rgb_predicted = np.moveaxis(
                    test_rgb_predicted, -1, 0
                )  # 3 x 200 x W
                writer.add_image("Test Image", test_image_log, i)
                writer.add_image("Rendered Image", test_rgb_predicted, i)

                # Sample sdf and rgb in all world points
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
                ).to(device)
                # Normalize world points
                world_points = (world_points / config["scene"]["world_size"]) * 2 - 1
                # Flatten world points
                world_points = world_points.view(-1, 2)
                if method == "neus":
                    world_points.requires_grad_(True)
                # Split world points into batches
                world_batches = common.get_minibatches(
                    world_points, chunksize=config["training"]["chunksize"]
                )
                # Predict sdf and rgb for all world points
                world_sdf_predictions = []
                world_color_predictions = []
                world_gradients = []
                for batch in world_batches:
                    world_sdf_predictions.append(distance_model(batch))
                    world_gradients.append(
                        torch.autograd.grad(
                            outputs=world_sdf_predictions[-1],
                            inputs=batch,
                            grad_outputs=torch.ones_like(world_sdf_predictions[-1]),
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                    )
                    world_color_predictions.append(color_model(batch))
                world_sdf_flattened = torch.cat(world_sdf_predictions, dim=0)
                world_color_flattened = torch.cat(world_color_predictions, dim=0)
                world_unflattened_distance_shape = [
                    scene_generation.WORLD_SIZE,
                    scene_generation.WORLD_SIZE,
                ]
                world_sdf = world_sdf_flattened.view(world_unflattened_distance_shape)
                world_unflattened_color_shape = [
                    scene_generation.WORLD_SIZE,
                    scene_generation.WORLD_SIZE,
                ] + [3]
                world_color = world_color_flattened.view(world_unflattened_color_shape)
                world_color = world_color.permute(1, 0, 2)
                world_sdf = world_sdf.permute(1, 0)

                world_sdf = np.expand_dims(
                    world_sdf.detach().cpu().numpy(),
                    axis=0,
                )
                world_color = np.moveaxis(
                    world_color.detach().cpu().numpy(),
                    -1,
                    0,
                )

                # Log sdf and rgb in tensorboard
                neus.plot_sdf(world_sdf, writer, i)
                writer.add_image("RGB map", np.flip(world_color, axis=1), i)

            elif method == "nerf":
                # Save model
                torch.save(model.state_dict(), f"{output}/model.pth")
                # Get the radiance field from the model
                test_predictions = []
                for batch in test_batches:
                    test_predictions.append(model(batch))
                test_radiance_field_flattened = torch.cat(test_predictions, dim=0)

                test_unflattened_shape = list(test_query_points.shape[:-1]) + [4]
                test_radiance_field = test_radiance_field_flattened.view(
                    test_unflattened_shape
                )

                test_rgb_predicted, test_depth_predicted, _ = (
                    nerf.render_volume_density_2d(
                        test_radiance_field,
                        test_ray_o.to(device),
                        test_depth_values,
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
                world_points = (world_points / config["scene"]["world_size"]) * 2 - 1
                # Flatten world points
                world_points = world_points.view(-1, 2)
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
                # World radiance field contain values as radiancefield[i,j] = f(i,j)
                # We want to plot this with i being the horizontal coordinate and j the vertical coordinate
                # Thus we need to transpose the radiance field
                world_radiance_field = world_radiance_field.permute(1, 0, 2)

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
                test_rgb_predicted = test_rgb_predicted.detach().cpu().numpy()  # W x 3
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

    if method == "nerf":
        # After training is completed, save the model
        torch.save(model.state_dict(), f"{output}/model.pth")
        typer.echo("Training completed, model saved.")
        # Generate a circular video around the scene
        typer.echo("Generating circular video...")
        camera.generate_videos(
            scene,
            list(zip(c_xs, c_ys)),
            [model],
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
            [model],
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
            [model],
            nerf.nerf_2d,
            output,
            "zoom_in",
            config,
        )

    elif method == "neus":
        # After training is completed, save the model
        torch.save(distance_model.state_dict(), f"{output}/distance_model.pth")
        torch.save(s_model.state_dict(), f"{output}/s_model.pth")
        torch.save(color_model.state_dict(), f"{output}/color_model.pth")
        typer.echo("Training completed, model saved.")
        # Generate a circular video around the scene
        typer.echo("Generating circular video...")
        camera.generate_videos(
            scene,
            list(zip(c_xs, c_ys)),
            [distance_model, s_model, color_model],
            neus.neus_2d,
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
            [distance_model, s_model, color_model],
            neus.neus_2d,
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
            [distance_model, s_model, color_model],
            neus.neus_2d,
            output,
            "zoom_in",
            config,
        )


if __name__ == "__main__":
    app()
