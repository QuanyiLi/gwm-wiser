"""Where the rig's shared files live.

`assets/` and `config/` sit at the `gwm_hardware/` package root, not inside a
subpackage, because both experiment arms and the tiptop-tree installers read
them. Import these constants rather than recomputing `__file__`-relative paths
-- that is what made the 2026-08-19 subpackage split a one-line change per
consumer instead of a hunt.
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]      # droid/gwm_hardware
DROID_ROOT = PKG_ROOT.parent                        # droid/
REPO_ROOT = DROID_ROOT.parent                       # the gwm-wiser checkout

ASSETS = PKG_ROOT / "assets"                        # generated, gitignored
CONFIG = PKG_ROOT / "config"                        # versioned
DOCS = PKG_ROOT / "docs"

TIPTOP_ROOT = DROID_ROOT / "tiptop"                 # the pristine upstream worktree
CUTAMP_ASSETS = TIPTOP_ROOT / "cutamp/cutamp/robots/assets"
