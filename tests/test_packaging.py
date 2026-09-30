"""Build-only release checks: python -m unittest discover -s tests -p test_packaging.py.

Requires setuptools>=77 and packaging; does not install runtime dependencies.
"""

from email.parser import Parser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_documented_install_extra(self):
        readme = (ROOT / "README.md").read_text()
        extras = re.findall(r"pip install -e '\.\[([^]]+)\]'", readme)
        self.assertTrue(extras, "README must document an editable installation")
        for extra in extras:
            requirement = Requirement(f"gwm_wiser[{extra}]")
            self.assertTrue(requirement.extras <= {"wiser", "gwm-wiser"})

    def _build(self, hook):
        with tempfile.TemporaryDirectory(prefix="gwm-packaging-") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            paths = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=ROOT,
            ).decode().split("\0")
            for relative in set(paths):
                if not relative:
                    continue
                path = Path(relative)
                if path.parts[0] not in {"gwm_wiser", "real_data_train"} and relative not in {
                    "setup.py", "pyproject.toml", "README.md", "LICENSE"
                }:
                    continue
                destination = source / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, destination)
            output = Path(directory) / "dist"
            output.mkdir()
            with (source / "pyproject.toml").open("rb") as file:
                backend = tomllib.load(file)["build-system"]["build-backend"]
            result = subprocess.run(
                [sys.executable, "-c", (
                    "from importlib import import_module; "
                    f"backend = import_module({backend!r}); "
                    f"backend.{hook}({str(output)!r})"
                )],
                cwd=source,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheels = list(output.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                names = set(wheel.namelist())
                metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
                metadata = Parser().parsestr(wheel.read(metadata_path).decode())
            self.assertEqual(set(metadata.get_all("Provides-Extra")), {"wiser", "gwm-wiser"})
            requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
            self.assertTrue(requirements)
            self.assertTrue(all(requirement.marker is not None for requirement in requirements))
            self.assertFalse(any(req.marker.evaluate({"extra": ""}) for req in requirements))
            for extra, expected in {
                "wiser": {"mani_skill", "lerobot", "tensordict"},
                "gwm-wiser": {"mani_skill", "lerobot", "tensordict", "transformers", "scikit-learn"},
            }.items():
                selected = {req.name for req in requirements if req.marker.evaluate({"extra": extra})}
                self.assertEqual(selected, expected)
            return names

    def test_wheel_build_and_assets(self):
        names = self._build("build_wheel")
        for required in (
            "gwm_wiser/env/config_registry.py",
            "gwm_wiser/assets/configs/config_0.yaml",
            "gwm_wiser/assets/images/0_country/test_china.png",
            "gwm_wiser/assets/images/0_country/train_us.jpg",
            "gwm_wiser/assets/images/0_country/ref_exp.json",
        ):
            self.assertIn(required, names)
        self.assertFalse(any("/__pycache__/" in name or "/assets/stats/" in name for name in names))

    def test_editable_wheel_build(self):
        self._build("build_editable")


if __name__ == "__main__":
    unittest.main()
