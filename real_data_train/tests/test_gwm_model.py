"""Variable-length GWM wrapper: dynamic lengths, MSE training, 1620 parity."""

import torch
import torch.nn.functional as F

from gwm_wiser.models.gwm import GroundedWorldModel
from gwm_wiser.models.transformer import TransformerConfig
from real_data_train.gwm_model import VariableLenGWM

SMALL = dict(dim=64, ffn_dim=128, head_dim=16, n_layer=2, n_head=4, n_kv_head=2)


def _small_config():
    return TransformerConfig(**SMALL)


def test_forward_accepts_non_wiser_sequence_lengths():
    model = VariableLenGWM(config=_small_config(), output_dim=4096)
    for n_per_level in (90, 135, 540):
        x = torch.randn(2, 4 * n_per_level, 4096)
        out = model(x)
        assert out.shape == (2, 4 * n_per_level, 4096)


def test_mse_training_step_produces_gradients():
    model = VariableLenGWM(config=_small_config(), output_dim=4096)
    x = torch.randn(1, 4 * 90, 4096)
    target = torch.randn(1, 4 * 90, 4096)
    loss = F.mse_loss(model(x), target)
    loss.backward()
    assert model.input_proj.weight.grad is not None
    assert model.output_proj.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters())


def test_wrapper_state_dict_loads_strictly_into_canonical_model():
    wrapper = VariableLenGWM(config=_small_config(), output_dim=4096)
    canonical = GroundedWorldModel(config=_small_config(), output_dim=4096)
    canonical.load_state_dict(wrapper.state_dict(), strict=True)


def test_wrapper_matches_canonical_output_at_1620_tokens():
    torch.manual_seed(0)
    wrapper = VariableLenGWM(config=_small_config(), output_dim=4096)
    canonical = GroundedWorldModel(config=_small_config(), output_dim=4096)
    canonical.load_state_dict(wrapper.state_dict(), strict=True)
    wrapper.eval()
    canonical.eval()
    x = torch.randn(1, 1620, 4096)
    with torch.no_grad():
        torch.testing.assert_close(wrapper(x), canonical(x), rtol=0, atol=0)
