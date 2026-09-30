"""Load trusted GWM checkpoints, including checkpoints from the old package."""

import pickle


class _GWMUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("vla_align.models.transformer", "TransformerConfig"):
            module = "gwm_wiser.models.transformer"
        return super().find_class(module, name)


class _CheckpointPickle:
    """The pickle interface used by torch.load for ZIP and legacy checkpoints."""

    Unpickler = _GWMUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _GWMUnpickler(file, **kwargs).load()


def load_gwm_checkpoint(path, map_location="cpu"):
    """Return a trusted checkpoint without constructing a model.

    Older WISER releases saved TransformerConfig under the vla_align package.
    Only that class is remapped to its current gwm_wiser location. As with
    torch.load(weights_only=False), this must only load trusted checkpoints.
    """
    import torch

    return torch.load(
        path,
        map_location=map_location,
        pickle_module=_CheckpointPickle,
        weights_only=False,
    )
