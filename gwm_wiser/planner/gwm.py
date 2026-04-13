"""
GWM-Based Planner using Grounded World Model for trajectory embedding prediction.

This planner extends RetrievalBasedPlanner by using a trained GWM model to predict
trajectory embeddings instead of computing them from ground truth video frames.
"""

import torch

from gwm_wiser.models.gwm import GroundedWorldModel, ActionConditionedGWM
from gwm_wiser.models.transformer import TransformerConfig
from gwm_wiser.planner.retrieval import (
    RetrievalBasedPlanner,
    RetrievedTrajectory,
)
from gwm_wiser.utils.gwm_data import encode_trajectory, tensor_images_to_pil


class GWMBasedPlanner(RetrievalBasedPlanner):
    """
    A planner that uses the Grounded World Model (GWM) to predict trajectory embeddings.

    Instead of encoding ground truth video frames to get trajectory embeddings,
    this planner uses the GWM to predict embeddings from:
    - Current image embedding (from observation)
    - Action sequence (from retrieved trajectory)

    Usage:
        planner = GWMBasedPlanner(
            dataset_root="/path/to/dataset",
            model=embedder,
            gwm_checkpoint_path="/path/to/checkpoint.pt",
            ...
        )
        actions = planner.act(obs)
    """

    def __init__(
        self,
        dataset_root: str,
        embedder,
        gwm_checkpoint_path: str,
        k: int = 12,
        num_future_frames: int = 60,
        replan_horizon: int = 10,
        video_frame_subsample: int = 10,
        verbose: bool = False,
        device="cuda",
        dtype=torch.float32,
        robot: str = "panda",
    ):
        """
        Initialize the GWM-based planner.

        Args:
            dataset_root: Path to the lerobot dataset
            model: Qwen3-VL-Embedding model (embedder)
            gwm_checkpoint_path: Path to the trained GWM checkpoint
            k: Number of candidate trajectories to retrieve
            num_future_frames: Number of future frames per trajectory
            replan_horizon: Replan every N steps
            video_frame_subsample: Subsample factor for video frames
            verbose: Whether to log detailed retrieval information
        """
        # Initialize parent class
        super().__init__(
            dataset_root=dataset_root,
            model=embedder,
            k=k,
            num_future_frames=num_future_frames,
            replan_horizon=replan_horizon,
            video_frame_subsample=video_frame_subsample,
            verbose=verbose,
            robot=robot,
        )

        # Load GWM model
        print(f"Loading GWM model from {gwm_checkpoint_path}...")
        checkpoint = torch.load(
            gwm_checkpoint_path, map_location="cpu", weights_only=False
        )

        # Get config from checkpoint
        config = checkpoint.get("config")
        if config is None:
            # Fallback to default config if not saved
            config = TransformerConfig()
            print(
                "Warning: Using default TransformerConfig (config not found in checkpoint)"
            )

        # Initialize GWM model
        self.gwm_model = GroundedWorldModel(
            config=config,
            output_dim=4096,
        )

        # load compiled models
        model_state_dict = {}
        prefix = "_orig_mod."
        for k, v in checkpoint["model_state_dict"].items():
            new_key = k.removeprefix(prefix)
            model_state_dict[new_key] = v

        # Load weights
        self.gwm_model.load_state_dict(model_state_dict)
        self.gwm_model.eval()

        # Move to same device as embedder if possible
        self.dtype = dtype
        self.gwm_model = self.gwm_model.to(dtype=self.dtype, device=device)

        print(
            f"GWM model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})"
        )
        if "test_cos_sim" in checkpoint:
            print(f"  Test cosine similarity: {checkpoint['test_cos_sim']:.4f}")
        if "test_mse" in checkpoint:
            print(f"  Test MSE: {checkpoint['test_mse']:.4f}")

    def get_video_embedding(self, traj: RetrievedTrajectory, current_frame_image):
        # only segmentation image should be available, which is also available during the test time
        traj.images_main = None
        traj.images_wrist = None
        with torch.no_grad():
            seg_image = tensor_images_to_pil(traj.images_robot_state)
            seg_image[0] = current_frame_image

            inputs = [{"video": seg_image}]
            encoded = encode_trajectory(self.embedder, inputs)  # (4096,)
            predicted_embedding = self.gwm_model(encoded[None, ...].to(self.dtype))[
                0
            ].to(torch.bfloat16)
            f_1, f_2, f_3, video_latent = predicted_embedding.chunk(4, dim=0)
            deepstack_video_embeds = [
                f_1,
                f_2,
                f_3,
            ]  # TODO: what is the correct order?... need double check
            video_outputs = self.embedder.embed_video_latent(
                video_latent, deepstack_video_embeds, inputs
            )
            video_embedding = self.embedder.pooling_video_latent(video_outputs)[0]
        return video_embedding


class ActionConditionedGWMPlanner(RetrievalBasedPlanner):
    """
    A planner that uses the Action-Conditioned GWM to predict trajectory embeddings.

    Instead of encoding a 6-frame segmentation video, this planner:
    1. Encodes only the current frame through Qwen (single image)
    2. Uses the retrieved trajectory's action sequence
    3. Forwards through ActionConditionedGWM to predict trajectory embedding

    Usage:
        planner = ActionConditionedGWMPlanner(
            dataset_root="/path/to/dataset",
            embedder=embedder,
            gwm_checkpoint_path="/path/to/ac_checkpoint.pt",
            ...
        )
        actions = planner.act(obs)
    """

    def __init__(
        self,
        dataset_root: str,
        embedder,
        gwm_checkpoint_path: str,
        k: int = 12,
        num_future_frames: int = 60,
        replan_horizon: int = 10,
        video_frame_subsample: int = 10,
        verbose: bool = False,
        device="cuda",
        dtype=torch.float32,
        robot: str = "panda",
    ):
        # Initialize parent class
        super().__init__(
            dataset_root=dataset_root,
            model=embedder,
            k=k,
            num_future_frames=num_future_frames,
            replan_horizon=replan_horizon,
            video_frame_subsample=video_frame_subsample,
            verbose=verbose,
            robot=robot,
        )

        # Load ActionConditionedGWM model
        print(f"Loading Action-Conditioned GWM model from {gwm_checkpoint_path}...")
        checkpoint = torch.load(
            gwm_checkpoint_path, map_location="cpu", weights_only=False
        )

        config = checkpoint.get("config")
        if config is None:
            config = TransformerConfig()
            print(
                "Warning: Using default TransformerConfig (config not found in checkpoint)"
            )

        self.gwm_model = ActionConditionedGWM(
            config=config,
            output_dim=4096,
        )

        # Load compiled models
        model_state_dict = {}
        prefix = "_orig_mod."
        for k_name, v in checkpoint["model_state_dict"].items():
            new_key = k_name.removeprefix(prefix)
            model_state_dict[new_key] = v

        self.gwm_model.load_state_dict(model_state_dict)
        self.gwm_model.eval()

        self.dtype = dtype
        self.gwm_model = self.gwm_model.to(dtype=self.dtype, device=device)

        print(
            f"AC-GWM model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})"
        )
        if "test_cos_sim" in checkpoint:
            print(f"  Test cosine similarity: {checkpoint['test_cos_sim']:.4f}")
        if "test_mse" in checkpoint:
            print(f"  Test MSE: {checkpoint['test_mse']:.4f}")

    def get_video_embedding(self, traj: RetrievedTrajectory, current_frame_image):
        """
        Predict video embedding using current frame + actions through ActionConditionedGWM.
        """
        traj.images_main = None
        traj.images_wrist = None

        with torch.no_grad():
            # 1. Encode current frame as single image through Qwen
            inputs = [{"image": current_frame_image}]
            conversations = [
                self.embedder.format_model_input(image=current_frame_image)
            ]
            processed_inputs = self.embedder._preprocess_inputs(conversations)
            processed_inputs = {
                k_name: v.to(self.embedder.model.device)
                for k_name, v in processed_inputs.items()
            }

            outputs = self.embedder.forward(processed_inputs)
            last_hidden = outputs["last_hidden_state"][0]  # (seq_len, 4096)
            attention_mask = outputs["attention_mask"][0]  # (seq_len,)
            frame_emb = last_hidden[attention_mask.bool()]  # (N_img, 4096)
            frame_emb = frame_emb.unsqueeze(0).to(self.dtype)  # (1, N_img, 4096)

            # 2. Get action sequence from retrieved trajectory
            actions = (
                traj.actions.unsqueeze(0).to(self.dtype).to(frame_emb.device)
            )  # (1, 60, 8)

            # 3. Forward through ActionConditionedGWM
            predicted_embedding = self.gwm_model(frame_emb, actions)[0].to(
                torch.bfloat16
            )  # (1620, 4096)

            # 4. Split into deepstack chunks and get final embedding
            f_1, f_2, f_3, video_latent = predicted_embedding.chunk(4, dim=0)
            deepstack_video_embeds = [f_1, f_2, f_3]

            # Need the original video-style inputs for embed_video_latent
            # Use a dummy video input (single frame repeated) to get proper input_ids/attention_mask
            seg_images = tensor_images_to_pil(traj.images_robot_state)
            seg_images[0] = current_frame_image
            video_inputs = [{"video": seg_images}]

            video_outputs = self.embedder.embed_video_latent(
                video_latent, deepstack_video_embeds, video_inputs
            )
            video_embedding = self.embedder.pooling_video_latent(video_outputs)[0]

        return video_embedding
