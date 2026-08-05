"""RAT condition/target tensors -> preprocessed Qwen inputs with a pixel budget.

Reuses the unchanged gwm_wiser preprocessing path; the per-source
aspect-preserving pixel budget (ADR-0014) is injected through the existing
per-video ``min_pixels``/``max_pixels`` hooks that ``fetch_video`` reads.

Default budget: the WISER-scale per-frame window. Its upper edge equals the
WISER processed frame size (480x288 = 138240 px), which lands sources near
405 tokens per level.
"""

import torch

from gwm_wiser.utils.gwm_data import tensor_images_to_pil  # noqa: F401  (shared with WISER pipeline)

# WISER-scale defaults (ADR-0014): lower edge is the production per-frame
# minimum; upper edge is the WISER processed frame area.
DEFAULT_MIN_PIXELS = 131072
DEFAULT_MAX_PIXELS = 138240


def _apply_pixel_budget(conversation, min_pixels, max_pixels):
    for message in conversation:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                item["min_pixels"] = min_pixels
                item["max_pixels"] = max_pixels


def _preprocess_video(frames, preprocessor, min_pixels, max_pixels):
    conversation = preprocessor.format_model_input(
        video=tensor_images_to_pil(frames)
    )
    _apply_pixel_budget(conversation, min_pixels, max_pixels)
    inputs = preprocessor._preprocess_inputs([conversation])
    # Drop the singleton batch dim so DataLoader collation re-adds it.
    return {
        k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
        for k, v in inputs.items()
    }


def rat_to_qwen_inputs(
    condition: torch.Tensor,
    target: torch.Tensor,
    preprocessor,
    min_pixels: int = None,
    max_pixels: int = None,
) -> dict:
    min_pixels = DEFAULT_MIN_PIXELS if min_pixels is None else min_pixels
    max_pixels = DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels
    return {
        "qwen_current_inputs": _preprocess_video(
            condition, preprocessor, min_pixels, max_pixels
        ),
        "qwen_trajectory_gt": _preprocess_video(
            target, preprocessor, min_pixels, max_pixels
        ),
    }


def count_visual_tokens(inputs: dict) -> int:
    """Four-level concatenated visual token count from the produced grid.

    Per level the merged token count is prod(grid)/4; concatenating the three
    DeepStack levels plus the final level gives exactly prod(grid).
    """
    grid = inputs["video_grid_thw"]
    return int(torch.as_tensor(grid).reshape(-1, 3).prod(dim=-1).sum().item())
