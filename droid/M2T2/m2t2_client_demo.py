#!/usr/bin/env python3
"""Client script to predict grasps using the M2T2 server."""
import argparse
import numpy as np
import os
import pickle
import requests
import torch
from PIL import Image

from m2t2.dataset_utils import denormalize_rgb, depth_to_xyz, normalize_rgb
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.plot_utils import get_set_colors


def load_from_data_dir(data_dir):
    """Load point cloud from raw data directory (depth.npy, rgb.png format)."""
    print(f"Loading from data directory: {data_dir}")

    # Load metadata
    with open(f'{data_dir}/meta_data.pkl', 'rb') as f:
        meta_data = pickle.load(f)

    # Load and normalize RGB
    rgb_img = normalize_rgb(Image.open(f'{data_dir}/rgb.png')).permute(1, 2, 0)

    # Load depth and convert to XYZ
    depth = np.load(f'{data_dir}/depth.npy')
    xyz = torch.from_numpy(
        depth_to_xyz(depth, meta_data['intrinsics'])
    ).float()

    # Load segmentation
    seg = torch.from_numpy(np.array(Image.open(f'{data_dir}/seg.png')))
    label_map = meta_data['label_map']

    # Filter out invalid depth points
    valid_mask = depth > 0
    xyz, rgb_img, seg = xyz[valid_mask], rgb_img[valid_mask], seg[valid_mask]

    # Transform to world coordinates
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
    xyz_world = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    # Apply scene bounds if available
    if 'scene_bounds' in meta_data:
        bounds = meta_data['scene_bounds']
        within = (xyz_world[:, 0] > bounds[0]) & (xyz_world[:, 0] < bounds[3]) \
            & (xyz_world[:, 1] > bounds[1]) & (xyz_world[:, 1] < bounds[4]) \
            & (xyz_world[:, 2] > bounds[2]) & (xyz_world[:, 2] < bounds[5])
        xyz_world, rgb_img = xyz_world[within], rgb_img[within]

    xyz = xyz_world.numpy()

    # Denormalize RGB for visualization (0-1 range)
    rgb_tensor = rgb_img
    rgb_denorm = denormalize_rgb(rgb_tensor.T.unsqueeze(2)).squeeze(2).T
    rgb = rgb_denorm.numpy()

    print(f"Loaded {xyz.shape[0]} points from {data_dir}")

    return xyz, rgb


def predict_grasps_via_server(
    xyz, rgb, server_url, num_points=16384, num_runs=5,
    mask_thresh=0.2, apply_bounds=True
):
    """Send point cloud to server and get grasp predictions.

    Args:
        xyz: Point cloud coordinates (N, 3) as numpy array
        rgb: RGB colors (N, 3) as numpy array in [0, 1] range
        server_url: URL of the M2T2 server
        num_points: Number of points to sample for inference
        num_runs: Number of inference runs to aggregate
        mask_thresh: Confidence threshold for predicted grasps
        apply_bounds: Whether to apply spatial bounds filtering

    Returns:
        Dictionary with grasps, grasp_confidence, and grasp_contacts
    """
    print(f"Sending request to {server_url}...")

    # Prepare request payload
    payload = {
        "pointcloud": {
            "points": xyz.tolist(),
            "rgb": rgb.tolist()
        },
        "num_points": num_points,
        "num_runs": num_runs,
        "mask_thresh": mask_thresh,
        "apply_bounds": apply_bounds
    }

    # Send POST request
    try:
        response = requests.post(
            f"{server_url}/predict",
            json=payload,
            timeout=300  # 5 minute timeout for inference
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        raise

    result = response.json()

    print(f"Received {result['num_grasps']} predicted grasps")
    print(f"Server processed {result['num_points_processed']} points")

    # Convert lists back to torch tensors for visualization
    outputs = {
        'grasps': [torch.tensor(g, dtype=torch.float32) for g in result['grasps']],
        'grasp_confidence': [torch.tensor(c, dtype=torch.float32) for c in result['grasp_confidence']],
        'grasp_contacts': [torch.tensor(c, dtype=torch.float32) for c in result['grasp_contacts']]
    }

    return outputs, result['num_points_processed']


def visualize_results(xyz, rgb, outputs):
    """Visualize the point cloud and predicted grasps.

    Args:
        xyz: Point cloud coordinates (N, 3) as numpy array
        rgb: RGB colors (N, 3) as numpy array
        outputs: Dictionary with grasps, grasp_confidence, and grasp_contacts
    """
    print("\nVisualizing results...")
    print("Open http://127.0.0.1:7000/static/ in your browser to see the visualization")

    vis = create_visualizer()

    # Prepare RGB for visualization (convert to uint8)
    # RGB is already in 0-1 range from load_from_data_dir
    rgb_viz = (rgb * 255).astype('uint8')

    # Create camera frame
    cam_pose = np.eye(4)
    make_frame(vis, 'camera', T=cam_pose)

    # Visualize point cloud
    visualize_pointcloud(vis, 'scene', xyz, rgb_viz, size=0.005)

    # Visualize grasps
    total_grasps = 0
    for i, (grasps, conf, contacts, color) in enumerate(zip(
        outputs['grasps'],
        outputs['grasp_confidence'],
        outputs['grasp_contacts'],
        get_set_colors()
    )):
        num_grasps = grasps.shape[0]
        total_grasps += num_grasps
        print(f"  Object {i:02d}: {num_grasps} grasps predicted")

        # Visualize contact points with confidence colors
        conf_np = conf.numpy()
        conf_colors = (np.stack([
            1 - conf_np, conf_np, np.zeros_like(conf_np)
        ], axis=1) * 255).astype('uint8')
        visualize_pointcloud(
            vis, f"object_{i:02d}/contacts",
            contacts.numpy(), conf_colors, size=0.01
        )

        # Visualize grasp poses
        grasps_np = grasps.numpy()
        for j, grasp in enumerate(grasps_np):
            visualize_grasp(
                vis, f"object_{i:02d}/grasps/{j:03d}",
                grasp, color, linewidth=0.2
            )

    print(f"\nTotal grasps predicted: {total_grasps}")
    print("\nVisualization complete!")
    print("The visualization will remain open. Press Ctrl+C to exit.")


def main():
    parser = argparse.ArgumentParser(
        description='Client for M2T2 grasp prediction server'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default='sample_data/real_world/00',
        help='Path to data directory containing depth.npy and rgb.png (default: sample_data/real_world/00)'
    )
    parser.add_argument(
        '--server-url',
        type=str,
        default='http://localhost:8123',
        help='URL of the M2T2 server (default: http://localhost:8123)'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=16384,
        help='Number of points to sample (default: 16384)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=5,
        help='Number of inference runs to aggregate (default: 5)'
    )
    parser.add_argument(
        '--mask-thresh',
        type=float,
        default=0.4,
        help='Confidence threshold for predicted grasps (default: 0.4)'
    )
    parser.add_argument(
        '--no-bounds',
        action='store_true',
        help='Disable spatial bounds filtering'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization'
    )

    args = parser.parse_args()

    # Check server health
    print("=" * 80)
    print(f"Checking server health at {args.server_url}...")
    try:
        health_response = requests.get(f"{args.server_url}/health", timeout=5)
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"Server status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
        if not health_data['model_loaded']:
            print("ERROR: Server model not loaded!")
            return
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot connect to server at {args.server_url}")
        print(f"  {e}")
        print("\nMake sure the server is running:")
        print("  python server.py --checkpoint m2t2.pth --config config.yaml")
        return

    if not args.no_viz:
        print("\nIMPORTANT: Make sure meshcat-server is running in a separate terminal!")
        print("  Run: meshcat-server")
        print("  Then open: http://127.0.0.1:7000/static/ in your browser")
    print("=" * 80)
    print()

    # Load point cloud from data directory
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("  Provide a directory containing depth.npy, rgb.png, and meta_data.pkl")
        return

    print(f"Loading point cloud from data directory: {args.data_dir}...")
    xyz, rgb = load_from_data_dir(args.data_dir)

    # Predict grasps via server
    outputs, num_points_processed = predict_grasps_via_server(
        xyz=xyz,
        rgb=rgb,
        server_url=args.server_url,
        num_points=args.num_points,
        num_runs=args.num_runs,
        mask_thresh=args.mask_thresh,
        apply_bounds=not args.no_bounds
    )

    # Visualize if requested
    if not args.no_viz:
        # Use the processed point cloud for visualization
        # (apply the same bounds filtering locally for visualization)
        if not args.no_bounds:
            bounds_mask = (
                (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= 1.0) &
                (xyz[:, 1] >= -0.3) & (xyz[:, 1] <= 0.3) &
                (xyz[:, 2] >= -0.2) & (xyz[:, 2] <= 0.5)
            )
            xyz = xyz[bounds_mask]
            rgb = rgb[bounds_mask]

        visualize_results(xyz, rgb, outputs)
    else:
        print("\nSkipping visualization (--no-viz flag set)")
        print(f"Total grasps predicted: {sum(len(g) for g in outputs['grasps'])}")


if __name__ == '__main__':
    main()
