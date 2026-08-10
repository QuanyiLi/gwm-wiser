"""Variable-length training wrapper around the canonical GroundedWorldModel.

The wrapper subclasses the canonical model so learned module names and
parameter shapes are identical by construction (ADR-0007); only the forward
pass differs, generating the two-coordinate positions
``(feature_level, flattened_visual_index)`` dynamically for sequence length
``4 * visual_token_count``. At 405 tokens per level this reproduces the
canonical 1620-token position buffer exactly — and the exact-grid policy
(plan D-14) pins every training sample to that point, so training is always
bit-exact with the canonical model. The variable-length forward is retained
deliberately: it is what keeps the token-scale iteration knob (plan, deferred
list) a budget-only change instead of a model rewrite.
"""

import torch

from gwm_wiser.models.gwm import GroundedWorldModel


class VariableLenGWM(GroundedWorldModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        assert seq_len % 4 == 0, (
            f"sequence length {seq_len} is not 4 concatenated visual levels"
        )
        n_per_level = seq_len // 4

        tokens = self.input_proj(x)

        d1 = torch.arange(4, device=x.device)
        d2 = torch.arange(n_per_level, device=x.device)
        grid = torch.stack(torch.meshgrid(d1, d2, indexing="ij"), dim=-1)
        positions = grid.reshape(-1, 2).unsqueeze(0).expand(bs, -1, -1)

        output, _ = self.backbone(tokens, positions)
        return self.output_proj(output)


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def export_canonical(model, path, config, step: int, metadata: dict = None):
    """Save a checkpoint loadable strictly by the original fixed-1620 evaluator.

    Plan steps: instantiate the canonical model with its original position
    buffer, copy the trained parameters into it, save the canonical state
    dictionary with the ``config`` key always embedded, and verify strict
    loading with the same logic used by GWMBasedPlanner.
    """
    import transformers

    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    raw = getattr(raw, "module", raw)  # unwrap DDP
    state_dict = _strip_compile_prefix(
        {k: v.detach().cpu() for k, v in raw.state_dict().items()}
    )

    float_dtype = next(
        v.dtype for v in state_dict.values() if v.is_floating_point()
    )
    canonical = GroundedWorldModel(config=config, output_dim=raw.output_dim)
    canonical = canonical.to(float_dtype)
    canonical.load_state_dict(state_dict, strict=True)

    payload = {
        "step": step,
        "epoch": step,  # keep the key the evaluator prints
        "model_state_dict": canonical.state_dict(),
        "config": config,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metadata": metadata or {},
    }
    torch.save(payload, path)

    # Verify strict loading exactly as the evaluation loader would.
    load_canonical_like_planner(path)
    return path


def load_canonical_like_planner(path):
    """Replicate GWMBasedPlanner's checkpoint loading (planner/gwm.py)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"canonical checkpoint missing 'config': {path}")
    model = GroundedWorldModel(config=config, output_dim=4096)
    model_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        model_state_dict[k.replace("_orig_mod.", "")] = v
    float_dtype = next(
        v.dtype for v in model_state_dict.values() if v.is_floating_point()
    )
    model = model.to(float_dtype)
    model.load_state_dict(model_state_dict)
    return model, checkpoint
