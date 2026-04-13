"""
Retrieval-based planner using MSE state matching and Qwen3-VL-Embedding scoring.

This planner:
1. Retrieves candidate trajectories based on current state (MSE)
2. Scores them using Qwen3-VL-Embedding (future images vs task prompt)
3. Executes the best trajectory's actions for N steps, then re-queries
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import StandardScaler

from gwm_wiser.utils.gwm_data import tensor_images_to_pil, PaddedLeRobotDataset
from gwm_wiser.utils.lerobot import (
    obs_state_key,
    image_1_key,
    image_1_robot_state,
    action_key,
    wrist_image_key,
    task_key,
    image_1_segmentation_mask_key,
)


@dataclass
class RetrievedTrajectory:
    """Container for a retrieved trajectory."""

    dataset_index: int
    episode_index: int
    frame_index: int
    mse: float
    task: str

    # Multi-frame data
    states: torch.Tensor  # Shape: (num_frames, state_dim)
    actions: torch.Tensor  # Shape: (num_frames, action_dim)
    images_main: torch.Tensor  # Shape: (num_frames, C, H, W)
    images_robot_state: torch.Tensor  # Shape: (num_frames, C, H, W)
    images_robot_state_mask: Optional[torch.Tensor]  # Shape: (num_frames, 1, H, W)
    images_wrist: torch.Tensor  # Shape: (num_frames, C, H, W)

    # Padding mask
    is_padded: torch.Tensor  # Shape: (num_frames,)


class MSETrajectoryRetriever:
    """
    State-based trajectory retriever using Mean Squared Error (MSE).

    Given a query state, finds trajectories in the dataset that start from
    similar states and returns the next N timesteps of data.

    Example:
        >>> retriever = MSETrajectoryRetriever(
        ...     dataset_root="/path/to/lerobot_data",
        ...     num_future_frames=20,
        ... )
        >>> trajectories = retriever.retrieve(query_state, k=5)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        num_future_frames: int = 20,
        normalize: bool = True,
        repo_id: str = "unused",
        video_frame_subsample: int = 10,
    ):
        """
        Initialize the retriever and build the state index.

        Args:
            dataset_root: Path to the lerobot dataset directory
            num_future_frames: Number of future frames to retrieve per trajectory
            normalize: Whether to normalize states before computing MSE
            repo_id: Repository ID placeholder for local datasets
            video_frame_subsample: Subsample factor for video frames (images only).
                                   Actions/states are loaded at full resolution.
        """
        self.dataset_root = Path(dataset_root)
        self.num_future_frames = num_future_frames
        self.normalize = normalize
        self.video_frame_subsample = video_frame_subsample
        self.num_image_frames = (
            num_future_frames + video_frame_subsample - 1
        ) // video_frame_subsample
        self.repo_id = repo_id

        # Load dataset with delta_timestamps for future frames
        self.dataset = self._load_dataset()
        self.fps = self.dataset.fps

        # Build state index
        self.all_states, self.valid_indices, self.episode_indices = (
            self._build_state_index()
        )
        self.episode_indices = np.array(self.episode_indices)

        # Initialize scaler for normalization
        if normalize:
            self.scaler = StandardScaler()
            self.normalized_states = self.scaler.fit_transform(self.all_states)
        else:
            self.scaler = None
            self.normalized_states = self.all_states

        print("MSETrajectoryRetriever initialized:")
        print(f"  Dataset: {self.dataset_root}")
        print(f"  Total episodes: {self.num_episodes}")
        print(f"  Total frames: {len(self.dataset)}")
        print(f"  Valid states indexed: {len(self.all_states)}")
        print(f"  State dim: {self.all_states.shape[1]}")
        print(f"  Future frames per trajectory: {self.num_future_frames}")

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the dataset."""
        return self.dataset.num_episodes

    def _load_dataset(self) -> PaddedLeRobotDataset:
        """Load the dataset with delta_timestamps configured for future frames.

        Images are subsampled by video_frame_subsample factor while
        actions/states are loaded at full resolution for execution.
        """
        # Full resolution for actions/states (needed for execution)
        return PaddedLeRobotDataset(
            repo_id=self.repo_id,
            root=self.dataset_root,
            video_frame_subsample=self.video_frame_subsample,
            num_future_frames=self.num_future_frames,
        )

    def _build_state_index(self) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Build a state index from the dataset.

        Indexes ALL states in the dataset. For states near the end of an episode
        that don't have enough future frames, the trajectory will be padded with
        the last valid frame when retrieved (using dataset's *_is_pad keys).

        Returns:
            states: numpy array of shape (N, state_dim)
            valid_indices: list of dataset indices
            episode_indices: list of episode indices for each state
        """
        states = []
        valid_indices = []
        episode_indices = []

        for ep_idx in range(self.dataset.num_episodes):
            ep_info = self.dataset.meta.episodes[ep_idx]
            ep_start = ep_info["dataset_from_index"]
            ep_end = ep_info["dataset_to_index"]
            ep_length = ep_end - ep_start

            # Include ALL frames in the episode
            for frame_in_ep in range(ep_length):
                dataset_idx = ep_start + frame_in_ep
                state = self.dataset.hf_dataset[dataset_idx][obs_state_key]
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                states.append(state)
                valid_indices.append(dataset_idx)
                episode_indices.append(ep_idx)

        return np.stack(states).astype(np.float32), valid_indices, episode_indices

    def retrieve(
        self,
        query_state: np.ndarray,
        k: int = 5,
        mse_threshold: float = 0.04,
        verbose=False,
    ) -> List[RetrievedTrajectory]:
        """
        Retrieve k trajectories with states most similar to the query.

        For each episode in the dataset, finds the state with lowest MSE to the query.
        Then selects the top k episodes based on their best-match MSE.

        Args:
            query_state: numpy array of shape (state_dim,)
            k: number of trajectories to retrieve (must be <= num_episodes)
            mse_threshold: maximum MSE allowed; trajectories with MSE > threshold are filtered out

        Returns:
            List of RetrievedTrajectory objects sorted by MSE (lowest first)
        """
        assert k <= self.num_episodes, (
            f"k ({k}) must be <= num_episodes ({self.num_episodes}). "
            f"Cannot retrieve more trajectories than episodes in dataset."
        )

        # Normalize query if needed
        query = query_state.reshape(1, -1)
        if self.scaler is not None:
            query = self.scaler.transform(query)

        # Compute MSE with all indexed states
        squared_diff = (self.normalized_states - query) ** 2
        mse_values = np.mean(squared_diff, axis=1)

        # Find best matching state for each episode
        best_per_episode = {}  # ep_idx -> (local_idx, mse)
        for local_idx, (ep_idx, mse) in enumerate(
            zip(self.episode_indices, mse_values)
        ):
            if ep_idx not in best_per_episode or mse < best_per_episode[ep_idx][1]:
                best_per_episode[ep_idx] = (local_idx, mse)

        # Sort episodes by their best MSE and take top k
        sorted_episodes = sorted(best_per_episode.items(), key=lambda x: x[1][1])
        top_k_episodes = sorted_episodes[:k]

        # Build trajectory objects, filtering by threshold
        trajectories = []
        for ep_idx, (local_idx, mse) in top_k_episodes:
            # Skip if MSE exceeds threshold
            if mse > mse_threshold:
                continue

            dataset_idx = self.valid_indices[local_idx]
            sample = self.dataset[dataset_idx]  # PaddedLeRobotDataset handles padding

            seg = (
                sample[image_1_robot_state]
                if image_1_robot_state in sample
                else sample[image_1_key]
            )
            trajectory = RetrievedTrajectory(
                dataset_index=dataset_idx,
                episode_index=sample["episode_index"].item(),
                frame_index=sample["frame_index"].item(),
                mse=float(mse),
                task=sample[task_key],
                states=sample[obs_state_key],
                actions=sample[action_key],
                images_main=sample[image_1_key],
                images_robot_state=seg,
                images_robot_state_mask=sample.get(image_1_segmentation_mask_key, None),
                images_wrist=sample[wrist_image_key],
                is_padded=sample["action_is_pad"],
            )
            trajectories.append(trajectory)

        # Fallback: if all trajectories filtered out, use all without threshold
        if len(trajectories) == 0:
            if verbose:
                print(
                    f"Warning: All trajectories exceed MSE threshold ({mse_threshold}). Using all {k} trajectories."
                )
            for ep_idx, (local_idx, mse) in top_k_episodes:
                dataset_idx = self.valid_indices[local_idx]
                sample = self.dataset[dataset_idx]
                seg = (
                    sample[image_1_robot_state]
                    if image_1_robot_state in sample
                    else sample[image_1_key]
                )
                trajectory = RetrievedTrajectory(
                    dataset_index=dataset_idx,
                    episode_index=sample["episode_index"].item(),
                    frame_index=sample["frame_index"].item(),
                    mse=float(mse),
                    task=sample[task_key],
                    states=sample[obs_state_key],
                    actions=sample[action_key],
                    images_main=sample[image_1_key],
                    images_robot_state=seg,
                    images_robot_state_mask=sample.get(
                        image_1_segmentation_mask_key, None
                    ),
                    images_wrist=sample[wrist_image_key],
                    is_padded=sample["action_is_pad"],
                )
                trajectories.append(trajectory)

        return trajectories

    def get_state_at_index(self, idx: int) -> np.ndarray:
        """Get the state at a given index in the state index."""
        return self.all_states[idx]


@dataclass
class ScoredTrajectory:
    """Trajectory with its embedding score."""

    trajectory: RetrievedTrajectory
    score: float


class RetrievalBasedPlanner:
    """
    A planner that retrieves trajectories from a dataset and scores them
    using vision-language embeddings.

    Usage:
        planner = RetrievalBasedPlanner(env, dataset_root, ...)
        actions = planner.act(obs)  # Returns batched actions
        planner.reset()  # Reset internal state
    """

    def __init__(
        self,
        dataset_root: str,
        model,
        k: int = 5,
        num_future_frames: int = 60,
        replan_horizon: int = 10,
        video_frame_subsample: int = 10,
        verbose: bool = False,
        robot: str = "panda",
        **kwargs,
    ):
        """
        Initialize the retrieval-based planner.

        Args:
            env: The ManiSkill environment
            dataset_root: Path to the lerobot dataset
            model: Qwen3-VL-Embedding model
            k: Number of candidate trajectories to retrieve
            num_future_frames: Number of future frames per trajectory
            replan_horizon: Replan every N steps (receding horizon control)
            video_frame_subsample: Subsample factor for video frames (passed to retriever)
            verbose: Whether to log detailed retrieval information
            robot: Robot type ('panda' or 'xarm6')
        """
        assert len(kwargs) == 0
        self.k = k
        self.num_future_frames = num_future_frames
        self.replan_horizon = replan_horizon
        self.verbose = verbose
        self.robot = robot

        # Validate replan_horizon
        assert num_future_frames >= replan_horizon + 1, (
            f"num_future_frames ({num_future_frames}) must be >= replan_horizon + 1 ({replan_horizon + 1})"
        )
        assert 1 <= replan_horizon <= num_future_frames, (
            f"replan_horizon ({replan_horizon}) must be between 1 and num_future_frames ({num_future_frames})"
        )

        # Initialize retriever (video subsampling happens at dataset level)
        print(f"Loading trajectory retriever from {dataset_root}...")
        self.retriever = MSETrajectoryRetriever(
            dataset_root=dataset_root,
            num_future_frames=num_future_frames,
            normalize=True,
            video_frame_subsample=video_frame_subsample,
        )

        # Initialize embedder
        self.embedder = model

        # Cache for task prompt embedding
        self._task_embedding_cache = {}

        # Instructions for embedding
        # self.image_instruction = "Represent robot actions for the user retrieval."
        self.text_instruction = (
            "Retrieve the video which can best finish the manipulation task specified by the user, "
            "given the layout of the workspace and the current frame observation."
        )

        # Per-env state tracking (class-level variables)
        self.current_trajectories: dict = {}  # env_idx -> RetrievedTrajectory
        self.steps_in_trajectory: dict = {}  # env_idx -> int
        self.query_count: int = 0  # Total number of replan queries
        self._seen_step_zero: bool = False  # Track if we've seen elapsed_step == 0

    @staticmethod
    def _parse_grasp_prompt(task_prompt: str) -> str:
        """
        Parse grasp prompt from full task prompt.

        Task prompt format: "Pick up the {pick} and place it onto the {place}."
        Returns: "grasp {pick}"
        """
        import re

        match = re.search(r"Pick up the (.+?) and place", task_prompt)
        if match:
            pick_object = match.group(1)
            return f"Pick up the {pick_object} from the table"
        raise ValueError("Failed to extract the grasp action! Fall back")
        return "grasp cube"  # fallback

    @staticmethod
    def _parse_place_prompt(task_prompt: str) -> str:
        """
        Parse place prompt from full task prompt.

        Task prompt format: "Pick up the {pick} and place it onto the {place}."
        Returns: "place onto the {place}"
        """
        import re

        match = re.search(r"place it onto the (.+?)\.", task_prompt)
        if match:
            place_location = match.group(1)
            return f"Place the grasped object to the {place_location} on the table"
        raise ValueError("Failed to extract the placement action! Fall back")

    def reset(self):
        """
        Reset internal state for specified environments or all environments.

        Args:
            env_indices: List of environment indices to reset. If None, resets all.
        """
        self.current_trajectories = {}
        self.steps_in_trajectory = {}
        self.query_count = 0
        self._seen_step_zero = False  # Reset step zero tracking
        # Clear image+text embedding cache since first frame changes on reset
        self._task_embedding_cache = {}

    def _get_task_embedding(
        self,
        task_prompt: str,
        first_frame_image: Optional[Image.Image] = None,
        current_frame_image: Optional[Image.Image] = None,
    ) -> torch.Tensor:
        """
        Get or compute embedding for task prompt with optional frame images.

        Args:
            task_prompt: The text prompt to encode
            first_frame_image: Optional PIL image of the first frame to include in encoding
            current_frame_image: Optional PIL image of the current frame to include in encoding

        Note:
            Embeddings are cached only when current_frame_image is None (i.e., first frame only).
            When current_frame_image is provided, embeddings are computed fresh each time
            since the current observation changes.
        """
        # Only use cache when no current frame (first frame is static per episode)
        if current_frame_image is None and task_prompt in self._task_embedding_cache:
            return self._task_embedding_cache[task_prompt]

        # Compute embedding
        images = []
        if first_frame_image is not None:
            images.append(first_frame_image)
        if current_frame_image is not None:
            images.append(current_frame_image)
        inputs = [
            {
                "text": task_prompt,
                "image": None if len(images) == 0 else images,
                "instruction": self.text_instruction,
            }
        ]

        embedding = self.embedder.process(inputs)[0]

        # Only cache when no current frame image (otherwise it changes each call)
        if current_frame_image is None:
            self._task_embedding_cache[task_prompt] = embedding
        return embedding

    def score_trajectories(
        self,
        trajectories: List[RetrievedTrajectory],
        task_prompt: str,
        first_frame_image: Optional[Image.Image] = None,
        current_frame_image: Optional[Image.Image] = None,
        is_grasped: bool = False,
    ) -> tuple:
        """
        Score trajectories by comparing their future images to task prompt.

        Uses both a grasp prompt and a place prompt, averaging
        the similarity scores.

        Args:
            trajectories: List of retrieved trajectories
            task_prompt: The task description to match against
            first_frame_image: Optional PIL image of first frame to include in prompt encoding
            current_frame_image: Optional PIL image of current frame to include in prompt encoding
            is_grasped: Whether the object is currently grasped (affects scoring weights)

        Returns:
            Tuple of (List of ScoredTrajectory sorted by score, List of sub_scores dicts)
        """
        # Get embeddings for grasp and place prompts (with first frame and current frame if provided)
        grasp_prompt = self._parse_grasp_prompt(task_prompt)
        place_prompt = self._parse_place_prompt(task_prompt)
        grasp_embedding = self._get_task_embedding(
            grasp_prompt,
            first_frame_image=first_frame_image,
            current_frame_image=current_frame_image,
        )
        place_embedding = self._get_task_embedding(
            place_prompt,
            first_frame_image=first_frame_image,
            current_frame_image=current_frame_image,
        )

        grasp_scores = []
        place_scores = []

        for traj in trajectories:
            video_embedding = self.get_video_embedding(traj, current_frame_image)

            # Compute cosine similarity with grasp prompt
            grasp_similarity = F.cosine_similarity(
                grasp_embedding.unsqueeze(0), video_embedding.unsqueeze(0)
            ).item()
            grasp_scores.append(grasp_similarity)

            # Compute cosine similarity with place prompt
            place_similarity = F.cosine_similarity(
                place_embedding.unsqueeze(0), video_embedding.unsqueeze(0)
            ).item()
            place_scores.append(place_similarity)

        # Apply softmax normalization to each score type separately
        # This ensures grasp and place contribute equally to the final ranking
        grasp_probs = F.softmax(torch.tensor(grasp_scores), dim=0).tolist()
        place_probs = F.softmax(torch.tensor(place_scores), dim=0).tolist()

        # Compute weighted average of normalized scores based on grasp state
        # If grasped: weight place higher (0.7)
        # If not grasped: weight grasp higher (0.7)
        grasp_weight = 0 if is_grasped else 1
        place_weight = 1 if is_grasped else 0

        scored = []
        sub_scores = []
        for i, traj in enumerate(trajectories):
            avg_score = grasp_weight * grasp_probs[i] + place_weight * place_probs[i]

            scored.append(ScoredTrajectory(trajectory=traj, score=avg_score))
            sub_scores.append(
                {
                    "episode": traj.episode_index,
                    "frame": traj.frame_index,
                    "grasp_score": grasp_scores[i],  # Raw cosine similarity
                    "place_score": place_scores[i],  # Raw cosine similarity
                    "grasp_score_norm": grasp_probs[i],  # Softmax normalized
                    "place_score_norm": place_probs[i],  # Softmax normalized
                    "avg_score_norm": avg_score,  # Average of normalized scores
                }
            )

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        # Also sort sub_scores by avg_score descending
        sub_scores.sort(key=lambda x: x["avg_score_norm"], reverse=True)
        return scored, sub_scores

    def get_video_embedding(self, traj: RetrievedTrajectory, current_frame_image):
        gt_future = tensor_images_to_pil(traj.images_main)
        inputs = [{"video": gt_future}]  # use GT future as if we can predict the future
        video_latent, deepstack_video_embeds = self.embedder.encode_video_to_latent(
            inputs
        )
        video_outputs = self.embedder.embed_video_latent(
            video_latent, deepstack_video_embeds, inputs
        )
        video_embedding = self.embedder.pooling_video_latent(video_outputs)[0]
        return video_embedding

    def act(self, obs: dict) -> torch.Tensor:
        """
        Compute batched actions for all environments.

        Args:
            obs: Observation dict with keys like obs_state_key, "task", etc.
                 Shape of obs[obs_state_key]: (num_envs, state_dim)

        Returns:
            Batched actions tensor of shape (num_envs, action_dim)
        """
        # Check for missing reset() call
        if "elapsed_steps" in obs:
            has_step_zero = (obs["elapsed_steps"] == 0).any().item()
            if has_step_zero:
                if self._seen_step_zero:
                    raise RuntimeError(
                        "elapsed_steps == 0 detected more than once without reset(). "
                        "You must call planner.reset() when the environment resets."
                    )
                self._seen_step_zero = True

        # Get number of envs from obs shape
        num_envs = obs[obs_state_key].shape[0]
        actions_list = []

        queried = False

        # Process each env
        for env_idx in range(num_envs):
            # Initialize tracking for new envs
            if env_idx not in self.steps_in_trajectory:
                self.steps_in_trajectory[env_idx] = (
                    self.replan_horizon + 1
                )  # Force initial query

            # Check if we need to re-query (on reset or after horizon steps)
            needs_requery = self.steps_in_trajectory[env_idx] >= self.replan_horizon + 1

            if needs_requery:
                # Get current state for this env
                state = obs[obs_state_key][env_idx].cpu().numpy()

                # Get task prompt for this env
                task_bytes = obs["task"][env_idx]
                task_prompt = bytes(task_bytes.tolist()).decode("utf-8").rstrip("\x00")

                # Get first frame image if available
                first_frame_key = image_1_key + "_first_frame"
                first_frame_image = None
                if first_frame_key in obs:
                    # Convert tensor to PIL image
                    first_frame_tensor = obs[first_frame_key][
                        env_idx
                    ]  # (H, W, C) or (C, H, W)
                    if first_frame_tensor.dim() == 3:
                        if first_frame_tensor.shape[0] in [1, 3, 4]:  # (C, H, W)
                            first_frame_np = (
                                first_frame_tensor.permute(1, 2, 0).cpu().numpy()
                            )
                        else:  # (H, W, C)
                            first_frame_np = first_frame_tensor.cpu().numpy()
                        if first_frame_np.max() <= 1.0:
                            first_frame_np = (first_frame_np * 255).astype(np.uint8)
                        else:
                            first_frame_np = first_frame_np.astype(np.uint8)
                        first_frame_image = Image.fromarray(first_frame_np)

                # Get current frame image if available
                current_frame_image = None
                if image_1_key in obs:
                    # Convert tensor to PIL image
                    current_frame_tensor = obs[image_1_key][
                        env_idx
                    ]  # (H, W, C) or (C, H, W)
                    if current_frame_tensor.dim() == 3:
                        if current_frame_tensor.shape[0] in [1, 3, 4]:  # (C, H, W)
                            current_frame_np = (
                                current_frame_tensor.permute(1, 2, 0).cpu().numpy()
                            )
                        else:  # (H, W, C)
                            current_frame_np = current_frame_tensor.cpu().numpy()
                        if current_frame_np.max() <= 1.0:
                            current_frame_np = (current_frame_np * 255).astype(np.uint8)
                        else:
                            current_frame_np = current_frame_np.astype(np.uint8)
                        current_frame_image = Image.fromarray(current_frame_np)

                # Determine if object is grasped based on last robot state dim
                last_dim_val = state[-1]
                if "xarm6" in self.robot:
                    is_grasped = last_dim_val < -0.1
                elif "panda" in self.robot:
                    is_grasped = last_dim_val < 0.02
                else:
                    raise ValueError(f"Unknown robot: {self.robot}")

                # Retrieve candidate trajectories
                trajectories = self.retriever.retrieve(
                    state, k=self.k, verbose=self.verbose
                )

                # Score and select best trajectory (with first frame + current frame context and grasp state)
                scored, sub_scores = self.score_trajectories(
                    trajectories,
                    task_prompt,
                    first_frame_image,
                    current_frame_image,
                    is_grasped=is_grasped,
                )
                self.current_trajectories[env_idx] = scored[0].trajectory
                self.steps_in_trajectory[env_idx] = 1

                # Log scored results
                if self.verbose:
                    if not queried:
                        self.query_count += 1
                        queried = True
                    print(
                        f"\n[Query #{self.query_count}] [Env {env_idx}] Scored trajectories "
                        f"(replan horizon={self.replan_horizon}, is_grasped={is_grasped})"
                    )
                    print("=" * 90)
                    print(
                        f"{'Rank':<5} {'Episode':<8} {'Frame':<7} {'Grasp':<10} {'Place':<10} {'Avg Score':<10}"
                    )
                    print("-" * 90)
                    for i, s in enumerate(sub_scores):
                        marker = ">>>" if i == 0 else "   "
                        print(
                            f"{marker}[{i + 1:<2}] {s['episode']:<8} {s['frame']:<7} "
                            f"{s['grasp_score_norm']:.4f}       {s['place_score_norm']:.4f}       {s['avg_score_norm']:.4f}"
                        )

            # Get action for current step
            action = self.current_trajectories[env_idx].actions[
                self.steps_in_trajectory[env_idx]
            ]
            self.steps_in_trajectory[env_idx] += 1
            actions_list.append(action)

        # Stack actions
        actions = (
            torch.stack(actions_list, dim=0)
            .to(obs[obs_state_key].device)
            .to(torch.float32)
        )

        return actions
