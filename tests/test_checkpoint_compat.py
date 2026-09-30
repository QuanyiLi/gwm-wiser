"""Checkpoint metadata regression tests; no model or public weights are loaded.

Run with ``python -m unittest discover -s tests -p test_checkpoint_compat.py``.
When PyTorch is absent, minimal import stubs allow the real TransformerConfig
definition to load. The torch.load integration test is then skipped.
"""

import importlib
import io
import pickle
import sys
import types
import unittest
import zipfile
from unittest import mock

from gwm_wiser.utils.checkpoint import _CheckpointPickle, load_gwm_checkpoint


try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


def _config_module():
    if torch is not None:
        return importlib.import_module("gwm_wiser.models.transformer")

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.nn = types.ModuleType("torch.nn")
    torch_stub.nn.Module = object
    torch_stub.nn.functional = types.ModuleType("torch.nn.functional")
    with mock.patch.dict(sys.modules, {
        "torch": torch_stub,
        "torch.nn": torch_stub.nn,
        "torch.nn.functional": torch_stub.nn.functional,
    }):
        module = importlib.import_module("gwm_wiser.models.transformer")
    return module


config_module = _config_module()
TransformerConfig = config_module.TransformerConfig


def _checkpoint_pickle(module, name="TransformerConfig"):
    # A small protocol-2 checkpoint with config.dim=512 and config.n_layer=7.
    # Literal GLOBAL opcodes keep this fixture independent of module aliases.
    return (
        b"\x80\x02}X\x06\x00\x00\x00config"
        + f"c{module}\n{name}\n".encode("ascii")
        + b")\x81}(X\x03\x00\x00\x00dimM\x00\x02"
        + b"X\x07\x00\x00\x00n_layerK\x07ubs."
    )


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {config_module.__name__: config_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_legacy_config_restores_current_class_and_fields(self):
        payload = _checkpoint_pickle("vla_align.models.transformer")
        with self.assertRaisesRegex(ModuleNotFoundError, "vla_align"):
            pickle.loads(payload)

        checkpoint = _CheckpointPickle.Unpickler(io.BytesIO(payload)).load()

        self.assertIs(type(checkpoint["config"]), TransformerConfig)
        self.assertEqual(checkpoint["config"].dim, 512)
        self.assertEqual(checkpoint["config"].n_layer, 7)
        self.assertEqual(checkpoint["config"].ffn_dim, TransformerConfig().ffn_dim)
        self.assertNotIn("vla_align", sys.modules)

    def test_current_checkpoint_is_unchanged(self):
        original = {
            "config": TransformerConfig(dim=384, n_layer=3),
            "model_state_dict": {"weight": [1, 2, 3]},
            "epoch": 8,
        }
        payload = pickle.dumps(original, protocol=2)
        loaded = _CheckpointPickle.Unpickler(io.BytesIO(payload)).load()
        self.assertEqual(loaded, original)

    def test_other_legacy_globals_are_not_remapped(self):
        for module, name in (
            ("vla_align.models.transformer", "OtherConfig"),
            ("vla_align.models.other", "TransformerConfig"),
        ):
            with self.subTest(module=module, name=name):
                with self.assertRaisesRegex(ModuleNotFoundError, "vla_align"):
                    _CheckpointPickle.Unpickler(
                        io.BytesIO(_checkpoint_pickle(module, name))
                    ).load()

    def test_standard_globals_are_delegated(self):
        original = {"values": {1, 2, 3}, "value": complex(2, 3)}
        payload = pickle.dumps(original, protocol=2)
        self.assertEqual(_CheckpointPickle.load(io.BytesIO(payload)), original)

    def test_load_interface_supports_legacy_streams(self):
        payload = _checkpoint_pickle("vla_align.models.transformer")
        config = _CheckpointPickle.load(io.BytesIO(payload), encoding="utf-8")["config"]
        self.assertIs(type(config), TransformerConfig)
        self.assertEqual(config.dim, 512)

    def test_public_helper_passes_torch_load_options(self):
        payload = _checkpoint_pickle("vla_align.models.transformer")

        def deserialize(file, *, map_location, pickle_module, weights_only):
            self.assertEqual(map_location, {"cuda:0": "cpu"})
            self.assertFalse(weights_only)
            self.assertIsInstance(pickle_module.__name__, str)
            return pickle_module.Unpickler(file).load()

        torch_stub = types.ModuleType("torch")
        torch_stub.load = mock.Mock(side_effect=deserialize)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            result = load_gwm_checkpoint(
                io.BytesIO(payload), map_location={"cuda:0": "cpu"}
            )
        self.assertEqual(result["config"].dim, 512)
        torch_stub.load.assert_called_once()

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_zip_and_legacy_checkpoint_roundtrip(self):
        original = {
            "config": TransformerConfig(dim=512, n_layer=7),
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        }
        current = b"cgwm_wiser.models.transformer\nTransformerConfig\n"
        legacy = b"cvla_align.models.transformer\nTransformerConfig\n"
        for use_zip in (True, False):
            for old_namespace in (False, True):
                with self.subTest(use_zip=use_zip, old_namespace=old_namespace):
                    buffer = io.BytesIO()
                    torch.save(
                        original,
                        buffer,
                        pickle_protocol=2,
                        _use_new_zipfile_serialization=use_zip,
                    )
                    if old_namespace:
                        if use_zip:
                            updated = io.BytesIO()
                            with zipfile.ZipFile(buffer) as source:
                                with zipfile.ZipFile(updated, "w") as target:
                                    for entry in source.infolist():
                                        data = source.read(entry)
                                        if entry.filename.endswith("/data.pkl"):
                                            self.assertIn(current, data)
                                            data = data.replace(current, legacy)
                                        target.writestr(entry, data)
                            buffer = updated
                        else:
                            self.assertIn(current, buffer.getvalue())
                            buffer = io.BytesIO(buffer.getvalue().replace(current, legacy))
                        buffer.seek(0)
                        with self.assertRaisesRegex(ModuleNotFoundError, "vla_align"):
                            torch.load(buffer, map_location="cpu", weights_only=False)
                    buffer.seek(0)
                    result = load_gwm_checkpoint(buffer)
                    self.assertEqual(result["config"], original["config"])
                    torch.testing.assert_close(
                        result["model_state_dict"]["weight"],
                        original["model_state_dict"]["weight"],
                    )


if __name__ == "__main__":
    unittest.main()
