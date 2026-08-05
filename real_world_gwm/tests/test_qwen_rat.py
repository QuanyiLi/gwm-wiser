"""RAT -> Qwen preprocessed inputs through the exact production path.

Uses the real Qwen3VLProcessor from the HF cache (no model weights, no GPU).
Expected grids are independent worked examples of the documented resize chain.
"""

import pytest
import torch

from real_world_gwm.qwen_rat import (
    count_visual_tokens,
    rat_to_qwen_inputs,
)

MODEL = "Qwen/Qwen3-VL-Embedding-8B"


@pytest.fixture(scope="session")
def preprocessor():
    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLPreprocessor

    return Qwen3VLPreprocessor(MODEL)


def _rat(h, w):
    torch.manual_seed(0)
    frames = torch.rand(6, 3, h, w)
    return frames, frames.clone()


def test_wiser_shape_input_lands_at_wiser_scale_preserving_aspect(preprocessor):
    # Budget window is applied per-frame (factor 64) then video-level
    # (factor 32): 224x448 min-bounces to exactly 256x512 = 131072 px.
    condition, target = _rat(224, 448)
    out = rat_to_qwen_inputs(condition, target, preprocessor)
    grid = out["qwen_trajectory_gt"]["video_grid_thw"].squeeze().tolist()
    assert grid == [3, 16, 32]
    assert count_visual_tokens(out["qwen_trajectory_gt"]) == 1536
    assert count_visual_tokens(out["qwen_current_inputs"]) == 1536


def test_vga_source_under_default_budget_lands_near_wiser_scale(preprocessor):
    # 480x640: per-frame max-clamp to 320x384, video-level min-bounce to
    # 352x416 -> (3, 22, 26) = 1716 tokens, near the 1620 WISER scale.
    condition, target = _rat(480, 640)
    out = rat_to_qwen_inputs(condition, target, preprocessor)  # default budget
    grid = out["qwen_trajectory_gt"]["video_grid_thw"].squeeze().tolist()
    assert grid == [3, 22, 26]
    assert count_visual_tokens(out["qwen_trajectory_gt"]) == 1716


def test_budget_override_restores_native_scale(preprocessor):
    condition, target = _rat(480, 640)
    out = rat_to_qwen_inputs(
        condition, target, preprocessor, min_pixels=131072, max_pixels=786432
    )
    grid = out["qwen_trajectory_gt"]["video_grid_thw"].squeeze().tolist()
    assert grid == [3, 32, 40]  # 640x512 native-snap -> 3840 tokens (> 2048 accepted)
    assert count_visual_tokens(out["qwen_trajectory_gt"]) == 3840


def test_dataloader_route_matches_inference_route_exactly(preprocessor):
    """No preprocessing inconsistency between training and gwm_eval-style inference.

    Route A: what the dataloader precomputes (rat_to_qwen_inputs), re-batched
    exactly as train.py feeds encode_video_to_latent(**tensors).
    Route B: what Qwen3VLEmbedder.encode_video_to_latent(inputs=[...]) would
    preprocess internally from the same PIL frames under the same pixel policy.
    """
    import torch as _torch

    from real_world_gwm.qwen_rat import (
        DEFAULT_MAX_PIXELS,
        DEFAULT_MIN_PIXELS,
        _apply_pixel_budget,
        tensor_images_to_pil,
    )

    torch.manual_seed(3)
    frames = torch.rand(6, 3, 480, 640)

    route_a = rat_to_qwen_inputs(frames, frames, preprocessor)["qwen_trajectory_gt"]
    route_a = {k: v.unsqueeze(0) for k, v in route_a.items()}  # train.py re-batching

    conversation = preprocessor.format_model_input(video=tensor_images_to_pil(frames))
    _apply_pixel_budget(conversation, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    route_b = preprocessor._preprocess_inputs([conversation])

    assert set(route_a) == set(dict(route_b))
    for k in route_a:
        a, b = route_a[k], route_b[k]
        # pixel_values_videos carries no batch dim from the processor, so
        # re-batching adds a leading singleton; the vision tower flattens it
        # (the unchanged WISER PaddedLeRobotDataset pipeline does the same).
        if a.dim() == b.dim() + 1 and a.shape[0] == 1:
            a = a[0]
        assert _torch.equal(a, b), f"mismatch in {k}"


def test_inputs_are_unbatched_tensors_for_dataloader_collation(preprocessor):
    condition, target = _rat(224, 448)
    out = rat_to_qwen_inputs(condition, target, preprocessor)
    for group in ("qwen_trajectory_gt", "qwen_current_inputs"):
        for k, v in out[group].items():
            assert isinstance(v, torch.Tensor), (group, k)
        assert out[group]["input_ids"].ndim == 1
