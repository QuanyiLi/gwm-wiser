#!/usr/bin/env python3
"""FastAPI server for M2T2 grasp prediction."""
import argparse
from typing import List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from omegaconf import OmegaConf

from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu


# Global model instance
model = None
cfg = None


class PointCloud(BaseModel):
    """Input point cloud data."""
    points: List[List[float]] = Field(
        ...,
        description="Point cloud as list of [x, y, z] coordinates"
    )
    rgb: Optional[List[List[float]]] = Field(
        None,
        description="Optional RGB colors as list of [r, g, b] values in [0, 1] range"
    )


class GraspPredictionRequest(BaseModel):
    """Request for grasp prediction."""
    pointcloud: PointCloud
    num_points: int = Field(
        16384,
        description="Number of points to sample for inference"
    )
    num_runs: int = Field(
        5,
        description="Number of inference runs to aggregate predictions"
    )
    mask_thresh: float = Field(
        0.2,
        description="Confidence threshold for predicted grasps"
    )
    apply_bounds: bool = Field(
        True,
        description="Apply spatial bounds filtering (x: [0, 1], y: [-0.3, 0.3], z: [-0.2, 0.5])"
    )


class GraspPredictionResponse(BaseModel):
    """Response containing predicted grasps."""
    num_grasps: int
    grasps: List[List[List[List[float]]]]  # List of objects, each with list of 4x4 transformation matrices
    grasp_confidence: List[List[float]]  # List of confidence scores per object
    grasp_contacts: List[List[List[float]]]  # List of contact points per object
    num_points_processed: int


app = FastAPI(
    title="M2T2 Grasp Prediction Server",
    description="Server for predicting grasps on point clouds using M2T2",
    version="1.0.0"
)


def normalize_rgb(rgb):
    """Normalize RGB values."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return (rgb - mean) / std


def prepare_data_from_pointcloud(
    xyz: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    apply_bounds: bool = True
):
    """Prepare data dictionary from point cloud in the format expected by M2T2.

    Args:
        xyz: Point cloud coordinates (N, 3)
        rgb: Optional RGB colors (N, 3) in [0, 1] range
        apply_bounds: Whether to apply spatial bounds filtering

    Returns:
        data: Dictionary with model inputs
        xyz: Filtered point cloud
        rgb: Filtered RGB colors
    """
    # Default white color if no RGB provided
    if rgb is None:
        rgb = torch.ones_like(xyz)

    # Apply spatial bounds to filter the point cloud
    if apply_bounds:
        bounds_mask = (
            (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= 1.0) &
            (xyz[:, 1] >= -0.3) & (xyz[:, 1] <= 0.3) &
            (xyz[:, 2] >= -0.2) & (xyz[:, 2] <= 0.5)
        )
        xyz = xyz[bounds_mask]
        rgb = rgb[bounds_mask]
        print(f"After applying bounds: {xyz.shape[0]} points remaining")

    # Normalize RGB
    rgb_normalized = normalize_rgb(rgb)

    # Center the point cloud
    xyz_centered = xyz - xyz.mean(dim=0)

    # Create camera pose (identity for world coordinates)
    cam_pose = torch.eye(4).float()

    # Prepare inputs (centered xyz + normalized rgb)
    inputs = torch.cat([xyz_centered, rgb_normalized], dim=1)

    # Create dummy segmentation (all points belong to the scene)
    seg = torch.ones(xyz.shape[0], dtype=torch.long)

    data = {
        'inputs': inputs,
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose,
        'task': 'pick',  # For grasp prediction
        # Dummy object inputs (not used for pick task but required by model)
        'object_inputs': torch.rand(1024, 6),
        'ee_pose': torch.eye(4).float(),
        'bottom_center': torch.zeros(3),
        'object_center': torch.zeros(3)
    }

    return data, xyz, rgb


def predict_grasps_from_pointcloud(
    xyz: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    num_points: int = 16384,
    num_runs: int = 5,
    mask_thresh: float = 0.2,
    apply_bounds: bool = True
):
    """Predict grasps from point cloud data.

    Args:
        xyz: Point cloud coordinates (N, 3)
        rgb: Optional RGB colors (N, 3) in [0, 1] range
        num_points: Number of points to sample for inference
        num_runs: Number of inference runs to aggregate
        mask_thresh: Confidence threshold for predicted grasps
        apply_bounds: Whether to apply spatial bounds filtering

    Returns:
        outputs: Dictionary with predicted grasps, confidence, and contacts
        num_points_processed: Number of points after filtering
    """
    global model, cfg

    if model is None:
        raise RuntimeError("Model not loaded. Server not initialized properly.")

    # Prepare data
    data, xyz_filtered, rgb_filtered = prepare_data_from_pointcloud(
        xyz, rgb, apply_bounds
    )

    inputs, xyz_data, seg = data['inputs'], data['points'], data['seg']
    obj_inputs = data['object_inputs']

    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': []
    }

    # Run multiple forward passes
    for run_idx in range(num_runs):
        # Sample points
        pt_idx = sample_points(xyz_data, num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz_data[pt_idx]
        data['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        data['object_inputs'] = obj_inputs[pt_idx]

        # Prepare batch
        data_batch = collate([data])
        to_gpu(data_batch)

        # Inference
        with torch.no_grad():
            cfg.eval.mask_thresh = mask_thresh
            cfg.eval.object_thresh = mask_thresh
            model_outputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_outputs)

        # Aggregate outputs
        for key in outputs:
            outputs[key].extend(model_outputs[key][0])

        print(f"  Run {run_idx+1}/{num_runs} complete")

    return outputs, xyz_filtered.shape[0]


@app.on_event("startup")
async def load_model():
    """Load the M2T2 model on server startup."""
    global model, cfg

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='weights/m2t2.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    # Parse args (will use defaults if running via uvicorn)
    args, _ = parser.parse_known_args()

    print(f"Loading config from {args.config}...")
    cfg = OmegaConf.load(args.config)

    print(f"Loading model from {args.checkpoint}...")
    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "M2T2 Grasp Prediction Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict grasps from point cloud",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=GraspPredictionResponse)
async def predict(request: GraspPredictionRequest):
    """Predict grasps from point cloud.

    Args:
        request: GraspPredictionRequest with point cloud data and parameters

    Returns:
        GraspPredictionResponse with predicted grasps
    """
    # Convert input to tensors
    xyz = torch.tensor(request.pointcloud.points, dtype=torch.float32)

    if xyz.shape[1] != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Point cloud must have shape (N, 3), got {xyz.shape}"
        )

    rgb = None
    if request.pointcloud.rgb is not None:
        rgb = torch.tensor(request.pointcloud.rgb, dtype=torch.float32)
        if rgb.shape != xyz.shape:
            raise HTTPException(
                status_code=400,
                detail=f"RGB shape {rgb.shape} must match points shape {xyz.shape}"
            )

    print(f"Received point cloud with {xyz.shape[0]} points")

    # Run prediction
    outputs, num_points_processed = predict_grasps_from_pointcloud(
        xyz=xyz,
        rgb=rgb,
        num_points=request.num_points,
        num_runs=request.num_runs,
        mask_thresh=request.mask_thresh,
        apply_bounds=request.apply_bounds
    )

    # Convert outputs to lists for JSON serialization
    total_grasps = sum(len(g) for g in outputs['grasps'])

    response = GraspPredictionResponse(
        num_grasps=total_grasps,
        grasps=[g.numpy().tolist() for g in outputs['grasps']],
        grasp_confidence=[c.numpy().tolist() for c in outputs['grasp_confidence']],
        grasp_contacts=[c.numpy().tolist() for c in outputs['grasp_contacts']],
        num_points_processed=num_points_processed
    )

    print(f"Predicted {total_grasps} grasps from {num_points_processed} points")

    return response


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(
        description='M2T2 Grasp Prediction Server'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8123,
        help='Port to bind the server to (default: 8123)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='weights/m2t2.pth',
        help='Path to model checkpoint (default: weights/m2t2.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Starting M2T2 Grasp Prediction Server")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    print("=" * 80)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == '__main__':
    main()
