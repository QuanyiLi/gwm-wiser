"""Point tiptop at this rig's config without editing the pristine tiptop tree.

`tiptop.config` reads `tiptop/config/tiptop.yml` and offers no override flag
or env var, and `tiptop-config` writes straight back to it -- so a hardware rig
cannot avoid touching that path. Replacing it with a symlink into
`gwm_hardware/config/` keeps the *content* versioned here: the only change
left in the tiptop worktree is a one-line symlink, and `tiptop-config` writes
through it into our copy.

Idempotent. The displaced upstream default is kept beside ours as
`tiptop.yml.upstream` the first time, so the stock values stay recoverable.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.install_rig_config
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.install_rig_config --restore
"""

import argparse
import shutil
from pathlib import Path

from gwm_hardware.common.paths import PKG_ROOT as HERE
RIG_CFG = HERE / "config/tiptop.yml"
UPSTREAM_BACKUP = HERE / "config/tiptop.yml.upstream"
TIPTOP_CFG = HERE.parent / "tiptop/tiptop/config/tiptop.yml"


def install() -> None:
    if not RIG_CFG.exists():
        raise SystemExit(f"rig config missing: {RIG_CFG}")
    if TIPTOP_CFG.is_symlink():
        target = TIPTOP_CFG.resolve()
        if target == RIG_CFG.resolve():
            print(f"already installed: {TIPTOP_CFG} -> {target}")
            return
        print(f"replacing existing symlink to {target}")
    elif TIPTOP_CFG.exists():
        if not UPSTREAM_BACKUP.exists():
            shutil.copy2(TIPTOP_CFG, UPSTREAM_BACKUP)
            print(f"saved upstream default to {UPSTREAM_BACKUP}")
    TIPTOP_CFG.unlink(missing_ok=True)
    TIPTOP_CFG.symlink_to(RIG_CFG)
    print(f"installed: {TIPTOP_CFG} -> {RIG_CFG}")


def restore() -> None:
    if not UPSTREAM_BACKUP.exists():
        raise SystemExit(f"no backup at {UPSTREAM_BACKUP}; restore with git instead")
    TIPTOP_CFG.unlink(missing_ok=True)
    shutil.copy2(UPSTREAM_BACKUP, TIPTOP_CFG)
    print(f"restored upstream default to {TIPTOP_CFG}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true",
                    help="put the upstream default back and drop the symlink")
    args = ap.parse_args()
    restore() if args.restore else install()


if __name__ == "__main__":
    main()
