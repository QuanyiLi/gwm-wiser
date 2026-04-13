import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from gwm_wiser.utils.lerobot import (
    image_1_key,
    image_1_robot_state,
    wrist_image_key,
    obs_state_key,
    action_key,
    image_1_segmentation_mask_key,
)


def encode_trajectory(embedder, inputs) -> torch.Tensor:
    """
    Encode trajectory images to embedding using Qwen3VL embedder.

    Args:
        embedder: Qwen3VLEmbedder instance
        inputs: List of dicts (raw) or Dict of tensors (preprocessed)

    Returns:
        Trajectory embedding tensor of shape (1620, 4096)
    """
    if isinstance(inputs, dict):
        video_latent, deepstack_video_embeds = embedder.encode_video_to_latent(**inputs)
    else:
        video_latent, deepstack_video_embeds = embedder.encode_video_to_latent(inputs)
    deepstack_video_embeds.append(video_latent)
    return torch.concatenate(deepstack_video_embeds, dim=0)


# Helper for embedding computation
def compute_embeddings_sequentially(embedder, batch_dict, test=False):
    """
    Compute embeddings for a batch of data sequentially to save memory.
    """
    cur_emb_list = []
    traj_emb_list = []

    assert "qwen_trajectory_gt" in batch_dict and "qwen_current_inputs" in batch_dict
    # Use preprocessed inputs
    traj_inputs_batch = batch_dict["qwen_trajectory_gt"]
    curr_inputs_batch = batch_dict["qwen_current_inputs"]
    # Assuming input_ids exists and is tensor
    batch_size = traj_inputs_batch["input_ids"].shape[0]

    # NOTE: for testing....
    if test:
        images_batch = batch_dict[image_1_key]  # (B, T, C, H, W)
        seg_images_batch = batch_dict[image_1_robot_state]  # (B, T, C, H, W)

    with torch.no_grad():
        for i in range(batch_size):
            # Slice inputs for this sample and add batch dim
            # We filter for tensors to be safe, though Qwen processor guarantees tensors
            t_in = {
                k: v[i].unsqueeze(0)
                for k, v in traj_inputs_batch.items()
                if isinstance(v, torch.Tensor)
            }
            c_in = {
                k: v[i].unsqueeze(0)
                for k, v in curr_inputs_batch.items()
                if isinstance(v, torch.Tensor)
            }

            # Move to device (embedder device)
            # Ideally dataset works on CPU, and we move to GPU here
            device = (
                embedder.model.device
                if hasattr(embedder, "model")
                else torch.device("cuda")
            )
            t_in = {k: v.to(device) for k, v in t_in.items()}
            c_in = {k: v.to(device) for k, v in c_in.items()}

            cur_emb = encode_trajectory(embedder, c_in)  # (4096,)
            traj_emb = encode_trajectory(embedder, t_in)  # (4096,)
            cur_emb_list.append(cur_emb)
            traj_emb_list.append(traj_emb)

            # NOTE: for testing....
            if test:
                imgs = images_batch[i]  # (T, C, H, W)
                segs = seg_images_batch[i]  # (T, C, H, W)
                segs[0] = imgs[0]
                cur_emb_1 = encode_trajectory(
                    embedder, [{"video": tensor_images_to_pil(segs)}]
                )  # (4096,)
                traj_emb_1 = encode_trajectory(
                    embedder, [{"video": tensor_images_to_pil(imgs)}]
                )  # (4096,)
                torch.testing.assert_close(cur_emb_1, cur_emb)
                torch.testing.assert_close(traj_emb_1, traj_emb)

    current_image_embeddings = torch.stack(cur_emb_list)  # (B, ...)
    trajectory_embeddings = torch.stack(traj_emb_list)  # (B, ...)

    return current_image_embeddings, trajectory_embeddings


def tensor_images_to_pil(images: torch.Tensor) -> list:
    """
    Convert tensor images to PIL format for embedder.

    Args:
        images: Tensor of shape (T, C, H, W) with values in [0, 1]

    Returns:
        List of PIL Images
    """
    pil_images = []
    for i in range(images.shape[0]):
        img = images[i].cpu().numpy()  # (C, H, W)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        pil_images.append(Image.fromarray(img))
    return pil_images


class PaddedLeRobotDataset(LeRobotDataset):
    """
    LeRobotDataset subclass that automatically replaces padded frames with
    the last valid frame in __getitem__, and optionally subsamples images.

    This is useful for trajectory retrieval where you want consistent frame
    counts even for frames near the end of episodes.
    """

    # Keys that should be subsampled (images for video embedding)
    IMAGE_KEYS = [
        image_1_robot_state,
        image_1_key,
        wrist_image_key,
        image_1_segmentation_mask_key,
    ]

    def __init__(
        self,
        repo_id,
        root,
        video_frame_subsample: int = 6,
        num_future_frames: int = 60,
        preprocess_qwen=False,
        preprocessor=None,
    ):
        """
        Initialize PaddedLeRobotDataset.

        Args:
            video_frame_subsample: Number of frames to subsample images to.
            num_future_frames: Total number of future frames loaded.
            preprocess_qwen: Whether to preprocess for Qwen.
            preprocessor: Qwen3VLPreprocessor instance (required if preprocess_qwen is True).
            *args, **kwargs: Passed to parent LeRobotDataset.__init__
        """
        self._preprocess = preprocess_qwen
        self.preprocessor = preprocessor
        self.video_frame_subsample = video_frame_subsample
        self.num_future_frames = num_future_frames
        delta_timestamps_future = [idx / 20.0 for idx in range(self.num_future_frames)]
        delta_timestamps = {
            image_1_key: delta_timestamps_future,
            wrist_image_key: delta_timestamps_future,
            image_1_robot_state: delta_timestamps_future,
            image_1_segmentation_mask_key: delta_timestamps_future,
            obs_state_key: delta_timestamps_future,
            action_key: delta_timestamps_future,
        }

        super().__init__(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps)

        # Precompute subsample indices for images
        self._subsample_indices = np.linspace(
            0, num_future_frames - 1, video_frame_subsample, dtype=int
        ).tolist()

    @staticmethod
    def _replace_padded_frames(
        data: torch.Tensor, is_pad: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace padded frames with the last valid frame.

        Args:
            data: Tensor of shape (T, ...) where T is number of frames
            is_pad: Boolean tensor of shape (T,) indicating which frames are padded

        Returns:
            Tensor with padded frames replaced by the last valid frame
        """
        if not is_pad.any():
            return data

        # Find the last valid frame index
        valid_mask = ~is_pad
        if not valid_mask.any():
            raise ValueError("All frames are padded, cannot replace with valid frame")

        # Get indices of valid frames and find the last one
        last_valid_idx = valid_mask.nonzero()[-1].item()
        last_valid_frame = data[last_valid_idx]

        # Replace padded frames with last valid frame
        result = data.clone()
        result[is_pad] = last_valid_frame
        return result

    def _preprocess_qwen(self, sample):
        """
        Preprocess sample for Qwen3VL embedding.

        Creates 'qwen_current_inputs' and 'qwen_trajectory_gt' in the sample.
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor required for _preprocess_qwen")

        # Get images and segmentations
        # They should already be subsampled by now if _getitem logic ran first
        imgs = sample[image_1_key]  # (T, C, H, W)
        segs = sample[image_1_robot_state]  # (T, C, H, W)

        # Create modified segmentation for current image embedding
        # (First frame is real image, rest are segmentation)
        # Note: We must clone to avoid modifying original 'segs' in place if it's used elsewhere
        segs_modified = segs.clone()
        segs_modified[0] = imgs[0]

        # Convert to PIL
        pil_imgs = tensor_images_to_pil(imgs)
        pil_segs = tensor_images_to_pil(segs_modified)

        # Format for Qwen
        # We need two separate inputs
        conversation_traj = self.preprocessor.format_model_input(
            video=pil_imgs,
        )

        conversation_curr = self.preprocessor.format_model_input(
            video=pil_segs,
        )

        # Preprocess to tensors
        # _preprocess_inputs returns a dict of tensors with batch dim (1, ...)
        inputs_traj = self.preprocessor._preprocess_inputs([conversation_traj])
        inputs_curr = self.preprocessor._preprocess_inputs([conversation_curr])

        # Remove batch dimension added by processing single list item
        for k, v in inputs_traj.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                inputs_traj[k] = v.squeeze(0)

        for k, v in inputs_curr.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                inputs_curr[k] = v.squeeze(0)

        return {"qwen_trajectory_gt": inputs_traj, "qwen_current_inputs": inputs_curr}

    def __getitem__(self, item):
        ret = self._getitem(item)
        if self._preprocess:
            return self._preprocess_qwen(ret)
        else:
            return ret

    def _getitem(self, idx: int) -> dict:
        """
        Get a sample with padded frames automatically replaced and images subsampled.

        Overrides parent __getitem__ to:
        1. Apply padding replacement on all keys with *_is_pad masks
        2. Subsample image keys to video_frame_subsample frames
        """
        sample = super().__getitem__(idx)

        # Find all keys that have padding masks and apply replacement
        keys_to_process = []
        for key in sample.keys():
            if key.endswith("_is_pad"):
                data_key = key[:-7]  # Remove '_is_pad' suffix
                if data_key in sample:
                    keys_to_process.append((data_key, key))

        # Apply padding replacement
        for data_key, pad_key in keys_to_process:
            sample[data_key] = self._replace_padded_frames(
                sample[data_key], sample[pad_key]
            )

        # Subsample images to video_frame_subsample frames
        for image_key in self.IMAGE_KEYS:
            if image_key in sample:
                sample[image_key] = sample[image_key][self._subsample_indices]
            # Also subsample the corresponding is_pad mask
            pad_key = f"{image_key}_is_pad"
            if pad_key in sample:
                sample[pad_key] = sample[pad_key][self._subsample_indices]

        return sample


class AugmentedPaddedDataset(PaddedLeRobotDataset):
    def __init__(
        self,
        *args,
        texture_aug_prob: float = 0.5,
        jitter_prob: float = 0.5,
        flip_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Augmentation hyperparameters
        self.jitter_prob = jitter_prob
        self.flip_prob = flip_prob
        self.texture_aug_prob = texture_aug_prob
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def _getitem(self, idx):
        sample = super()._getitem(idx)

        # Apply augmentation to image_1 and image_1_segmentation if they exist
        assert (
            image_1_key in sample
            and image_1_robot_state in sample
            and image_1_segmentation_mask_key in sample
        )

        # 1. Random Horizontal Flip
        if random.random() < self.flip_prob:
            sample[image_1_key] = T.functional.hflip(sample[image_1_key])
            sample[image_1_robot_state] = T.functional.hflip(
                sample[image_1_robot_state]
            )
            sample[image_1_segmentation_mask_key] = T.functional.hflip(
                sample[image_1_segmentation_mask_key]
            )

        img = sample[image_1_key]  # (T, C, H, W)
        robot_state_image = sample[image_1_robot_state]  # (T, C, H, W)
        segmentation_mask = (
            sample[image_1_segmentation_mask_key][:, 0] * 255
        )  # the last chanel is always the same
        segmentation_mask = segmentation_mask.to(torch.uint8)

        # 2. Color Jitter
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            self.color_jitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
        )

        # Apply to image
        for fn_id, b, c, s, h in [
            (fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        ]:
            transforms = []
            if brightness_factor is not None:
                transforms.append(
                    lambda x: T.functional.adjust_brightness(x, brightness_factor)
                )
            if contrast_factor is not None:
                transforms.append(
                    lambda x: T.functional.adjust_contrast(x, contrast_factor)
                )
            if saturation_factor is not None:
                transforms.append(
                    lambda x: T.functional.adjust_saturation(x, saturation_factor)
                )
            if hue_factor is not None:
                transforms.append(lambda x: T.functional.adjust_hue(x, hue_factor))

            # Reorder based on fn_idx
            # fn_idx is a list of indices like [2, 0, 1, 3] etc.
            # But get_params returns (fn_idx, b, c, s, h) where fn_idx is the list order?
            # Actually get_params returns (fn_idx, brightness_factor, ...). fn_idx is the order.

            # Let's map indices to the transform functions we created
            # 0: brightness, 1: contrast, 2: saturation, 3: hue
            ordered_transforms = []
            base_transforms = [
                lambda x: T.functional.adjust_brightness(x, brightness_factor),
                lambda x: T.functional.adjust_contrast(x, contrast_factor),
                lambda x: T.functional.adjust_saturation(x, saturation_factor),
                lambda x: T.functional.adjust_hue(x, hue_factor),
            ]

            for i in fn_idx:
                ordered_transforms.append(base_transforms[i])

            for t in ordered_transforms:
                img = t(img)

        sample[image_1_key] = img
        sample[image_1_robot_state] = robot_state_image

        return sample


class ActionConditionedPaddedDataset(PaddedLeRobotDataset):
    """
    Dataset for action-conditioned GWM training.

    Instead of creating a 6-frame segmentation video as input, this dataset:
    - Processes only the **first frame** (current observation) as a single image
    - Returns the raw action sequence (60, 8) for action conditioning
    - Still computes the same GT trajectory embedding (6-frame video)
    """

    def __init__(self, *args, flip_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_prob = flip_prob

    def _preprocess_qwen(self, sample):
        """
        Preprocess sample for action-conditioned GWM.

        Creates:
        - 'qwen_current_frame': single-image Qwen inputs for current observation
        - 'qwen_trajectory_gt': 6-frame video Qwen inputs for GT trajectory
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor required for _preprocess_qwen")

        # Get images for GT trajectory (6-frame video, same as standard GWM)
        imgs = sample[image_1_key]  # (T, C, H, W)

        # GT trajectory: all 6 real images as video
        pil_imgs = tensor_images_to_pil(imgs)
        conversation_traj = self.preprocessor.format_model_input(video=pil_imgs)
        inputs_traj = self.preprocessor._preprocess_inputs([conversation_traj])

        # Current frame: only the first image, processed as a single image
        pil_first_frame = pil_imgs[0]
        conversation_curr = self.preprocessor.format_model_input(image=pil_first_frame)
        inputs_curr = self.preprocessor._preprocess_inputs([conversation_curr])

        # Remove batch dimension
        for k, v in inputs_traj.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                inputs_traj[k] = v.squeeze(0)

        for k, v in inputs_curr.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                inputs_curr[k] = v.squeeze(0)

        return {
            "qwen_trajectory_gt": inputs_traj,
            "qwen_current_frame": inputs_curr,
            action_key: sample[action_key],  # (60, 8)
        }

    def _getitem(self, idx):
        sample = super()._getitem(idx)
        return sample


def encode_single_image(embedder, inputs) -> torch.Tensor:
    """
    Encode a single image to embedding using Qwen3VL embedder.

    For a single image, the processor creates pixel_values (not pixel_values_videos).
    We use the model's forward pass and extract the image features directly.

    Args:
        embedder: Qwen3VLEmbedder instance
        inputs: Dict of tensors (preprocessed single-image inputs)

    Returns:
        Image embedding tensor of shape (N_img, 4096)
    """
    outputs = embedder.forward(inputs)
    # For single image, the hidden states contain the image token embeddings
    # Extract the visual tokens from the last hidden state
    last_hidden = outputs["last_hidden_state"][0]  # (seq_len, 4096)
    attention_mask = outputs["attention_mask"][0]  # (seq_len,)

    # Return all attended tokens (includes text prompt tokens + image tokens)
    # The image tokens are the visual features we want
    # For consistency with the video path, return ALL tokens that are attended
    valid_tokens = last_hidden[attention_mask.bool()]
    return valid_tokens


def compute_embeddings_ac(embedder, batch_dict):
    """
    Compute embeddings for action-conditioned GWM training.

    Processes single-frame current observation and 6-frame GT trajectory.

    Returns:
        (current_frame_embs, gt_trajectory_embs) where:
        - current_frame_embs: (B, N_img, 4096) — stacked single-frame embeddings
        - gt_trajectory_embs: (B, 1620, 4096) — stacked GT trajectory embeddings
    """
    cur_emb_list = []
    traj_emb_list = []

    assert "qwen_trajectory_gt" in batch_dict and "qwen_current_frame" in batch_dict
    traj_inputs_batch = batch_dict["qwen_trajectory_gt"]
    curr_inputs_batch = batch_dict["qwen_current_frame"]
    batch_size = traj_inputs_batch["input_ids"].shape[0]

    with torch.no_grad():
        for i in range(batch_size):
            # GT trajectory: encode as video (same as standard GWM)
            t_in = {
                k: v[i].unsqueeze(0)
                for k, v in traj_inputs_batch.items()
                if isinstance(v, torch.Tensor)
            }
            device = (
                embedder.model.device
                if hasattr(embedder, "model")
                else torch.device("cuda")
            )
            t_in = {k: v.to(device) for k, v in t_in.items()}
            traj_emb = encode_trajectory(embedder, t_in)  # (1620, 4096)
            traj_emb_list.append(traj_emb)

            # Current frame: encode as single image
            c_in = {
                k: v[i].unsqueeze(0)
                for k, v in curr_inputs_batch.items()
                if isinstance(v, torch.Tensor)
            }
            c_in = {k: v.to(device) for k, v in c_in.items()}
            cur_emb = encode_single_image(embedder, c_in)  # (N_img, 4096)
            cur_emb_list.append(cur_emb)

    # Stack GT trajectories (all same shape)
    gt_trajectory_embs = torch.stack(traj_emb_list)  # (B, 1620, 4096)

    # Stack current frame embeddings - they should all be the same N_img
    # since images have the same resolution within a dataset
    current_frame_embs = torch.stack(cur_emb_list)  # (B, N_img, 4096)

    return current_frame_embs, gt_trajectory_embs
