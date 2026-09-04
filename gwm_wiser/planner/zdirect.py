"""
z-direct planner: the predictor outputs the pooled 4096-d clip embedding
directly, so the cost is -<z_hat, z_g> and the 8B readout (embed_video_latent +
pooling) is never applied to a prediction. Everything else (retrieval, k=12,
grasp/place prompt decomposition, softmax normalisation, replanning) is
inherited unchanged from RetrievalBasedPlanner via GWMBasedPlanner.
"""

import torch
import torch.nn.functional as F

from gwm_wiser.models.gwm import PooledGWM
from gwm_wiser.planner.gwm import GWMBasedPlanner
from gwm_wiser.planner.retrieval import RetrievedTrajectory
from gwm_wiser.utils.gwm_data import encode_trajectory, tensor_images_to_pil


class ZDirectPlanner(GWMBasedPlanner):
    def _build_gwm(self, config):
        return PooledGWM(config=config, output_dim=4096)

    def get_video_embedding(self, traj: RetrievedTrajectory, current_frame_image):
        traj.images_main = None
        traj.images_wrist = None
        with torch.no_grad():
            seg_image = tensor_images_to_pil(traj.images_robot_state)
            seg_image[0] = current_frame_image
            inputs = [{"video": seg_image}]
            encoded = encode_trajectory(self.embedder, inputs)  # (1620, 4096)
            z_hat = self.gwm_model(encoded[None, ...].to(self.dtype))[0]  # (4096,)
            # cosine scoring is scale-invariant; normalise for parity with the
            # oracle's L2-normalised pooled vector
            return F.normalize(z_hat.float(), dim=-1).to(torch.bfloat16)
