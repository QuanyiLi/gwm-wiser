"""Canonical checkpoint export: strict-loadable by the unchanged evaluator."""

import torch

from gwm_wiser.models.transformer import TransformerConfig
from real_world_gwm.gwm_model import (
    VariableLenGWM,
    export_canonical,
    load_canonical_like_planner,
)

SMALL = dict(dim=64, ffn_dim=128, head_dim=16, n_layer=2, n_head=4, n_kv_head=2)


def test_exported_checkpoint_strict_loads_with_planner_logic(tmp_path):
    config = TransformerConfig(**SMALL)
    wrapper = VariableLenGWM(config=config, output_dim=4096)
    path = tmp_path / "checkpoint.pt"
    export_canonical(wrapper, path, config=config, step=7)

    model, checkpoint = load_canonical_like_planner(path)
    assert checkpoint["config"] is not None  # never rely on the silent default
    assert checkpoint["step"] == 7

    wrapper.eval()
    model.eval()
    x = torch.randn(1, 1620, 4096)
    with torch.no_grad():
        torch.testing.assert_close(model(x), wrapper(x), rtol=0, atol=0)


def test_exported_checkpoint_records_versions_and_metadata(tmp_path):
    import transformers

    config = TransformerConfig(**SMALL)
    wrapper = VariableLenGWM(config=config, output_dim=4096)
    path = tmp_path / "checkpoint.pt"
    export_canonical(
        wrapper, path, config=config, step=0, metadata={"manifest_hash": "abc123"}
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["versions"]["transformers"] == transformers.__version__
    assert checkpoint["versions"]["torch"] == torch.__version__
    assert checkpoint["metadata"]["manifest_hash"] == "abc123"


def test_export_strips_compile_prefix(tmp_path):
    config = TransformerConfig(**SMALL)
    wrapper = VariableLenGWM(config=config, output_dim=4096)
    compiled = torch.compile(wrapper)
    path = tmp_path / "checkpoint.pt"
    export_canonical(compiled, path, config=config, step=0)
    model, _ = load_canonical_like_planner(path)
    assert not any(k.startswith("_orig_mod.") for k in model.state_dict())
