import sys

# The released GWM checkpoint pickles its config as
# ``vla_align.models.transformer.TransformerConfig`` (the pre-release package name).
# Alias it to the identical class in this repo so ``torch.load`` resolves it.
if "vla_align.models.transformer" not in sys.modules:
    import types

    from gwm_wiser.models import transformer as _transformer

    for _name in ("vla_align", "vla_align.models"):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    sys.modules["vla_align.models.transformer"] = _transformer
