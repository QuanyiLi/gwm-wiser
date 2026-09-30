# Loading the released checkpoints

Run these checks on the machine where you will run GWM, after following the [setup instructions](../README.md#wiser-setup). They load weights into CPU memory without starting a simulator, planning, or robot execution.

## Download

From the repository root, download the checkpoint for your workflow:

```bash
# WISER
hf download Shady0057/GWM checkpoint.pt --local-dir gwm_ckpt

# Real-data model for simulation and hardware
hf download Shady0057/GWM real_data/checkpoint.pt --local-dir gwm_ckpt
```

The two checkpoints serve different workflows. The [real-data model card](https://huggingface.co/Shady0057/GWM/blob/main/real_data/README.md) describes its architecture and training provenance.

Planning also requires the separate frozen encoder:

```bash
hf download Qwen/Qwen3-VL-Embedding-8B --local-dir qwen3-vl-embedding-8b
```

The encoder is not needed for the checkpoint-only check below. Set the encoder and checkpoint paths in your chosen launch recipe before running a planner.

## Check checkpoint loading

Use `load_gwm_checkpoint` for trusted GWM checkpoints. It preserves their configuration objects and maps the original WISER configuration class to its current package location. Like the existing planner, it uses Python pickle deserialization, so only load checkpoints from sources you trust.

```bash
python - gwm_ckpt/checkpoint.pt <<'PY'
import sys
from gwm_wiser.utils.checkpoint import load_gwm_checkpoint

for path in sys.argv[1:]:
    checkpoint = load_gwm_checkpoint(path, map_location="cpu")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state_dict")
    assert config is not None, "Checkpoint is missing its configuration"
    assert isinstance(state, dict) and state, "Checkpoint has no model weights"
    print(f"Loaded {path}: {len(state)} model tensors")
    print(f"Configuration: {type(config).__module__}.{type(config).__name__}")
PY
```

To check the real-data checkpoint, replace the argument with `gwm_ckpt/real_data/checkpoint.pt`. Both planners use this same compatibility loader. This check verifies deserialization; it does not execute the model or test the simulator and GPU dependencies.

For a strict architecture-and-weights check of the real-data checkpoint, its existing loader also constructs the model on CPU:

```python
from real_data_train.gwm_model import load_canonical_like_planner

model, checkpoint = load_canonical_like_planner("gwm_ckpt/real_data/checkpoint.pt")
print(type(model).__name__, checkpoint.get("step"))
```

See the [WISER entrypoints](../README.md#wiser-setup) and [robotics setup](../README.md#real-data-and-robotics-setup) for the environments needed to execute these models.
