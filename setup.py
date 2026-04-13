# gwm_wiser/setup.py

import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

# ---------------------------------------------------------------------------
# Guard: bare `pip install -e .` (or `pip install .`) must fail.
# Users must specify an extra: pip install -e .[wiser]  or  pip install -e .[gwm+wiser]
# ---------------------------------------------------------------------------

_ERROR_MSG = (
    "\n\n"
    "ERROR: You must specify an extras group to install this package.\n"
    "Available extras:\n"
    "  pip install -e '.[wiser]'       – WISER dataset and simulator (ManiSkill + LeRobot)\n"
    "  pip install -e '.[gwm+wiser]'   – WISER + GWM training/evaluation(adds transformers, scikit-learn, …)\n"
    "\n"
)


class _AbortInstall(install):
    """Abort bare `pip install .` with a helpful message."""

    def run(self):
        sys.stderr.write(_ERROR_MSG)
        sys.exit(1)


class _AbortDevelop(develop):
    """Abort bare `pip install -e .` with a helpful message."""

    def run(self):
        sys.stderr.write(_ERROR_MSG)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dependency groups
# ---------------------------------------------------------------------------

_wiser_requires = [
    "mani_skill>=3.0.0",
    "lerobot>=0.4.3",
    "tensordict",
]

_gwm_wiser_requires = _wiser_requires + [
    "transformers==4.57.6",
    "scikit-learn",
]

extras_require = {
    "wiser": _wiser_requires,
    "gwm+wiser": _gwm_wiser_requires,
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name="gwm_wiser",
    version="0.1.0",
    packages=find_packages(),
    # No base dependencies – everything lives in extras
    install_requires=[],
    extras_require=extras_require,
    # Override default install / develop to block bare installs
    cmdclass={
        "install": _AbortInstall,
        "develop": _AbortDevelop,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
