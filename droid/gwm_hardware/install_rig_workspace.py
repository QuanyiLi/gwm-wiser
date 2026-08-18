"""Point tiptop's workspace dispatch at this rig's obstacles.

`tiptop/workspace.py:workspace_cuboids()` sends `panda_robotiq` to
`fr3_workspace()` -- MIT LIS's bench geometry, which on this rig is both
hallucinated and incomplete. There is no config hook for it, so this rewrites
the one dispatch line; the geometry itself lives in `gwm_hardware/rig_workspace.py`
so the pristine tiptop worktree carries only a three-line diff.

Idempotent, keeps a `.orig`, and `--restore` reverts.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_rig_workspace
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_rig_workspace --verify
"""

import argparse
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "tiptop/tiptop/workspace.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARKER = "# --- patched by gwm_hardware.install_rig_workspace ---"

OLD = """    elif cfg.robot.type == "panda_robotiq":
        cuboids = fr3_workspace()"""
NEW = f"""    elif cfg.robot.type == "panda_robotiq":
        {MARKER}
        # This rig is not MIT LIS's bench; fr3_workspace() would invent
        # obstacles that are not here and omit the table edges that are.
        from gwm_hardware.rig_workspace import zhiwei_workspace

        cuboids = zhiwei_workspace()"""


def install() -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print("already patched")
        return
    if OLD not in text:
        raise SystemExit("tiptop/workspace.py does not match what this patch expects "
                         "-- upstream changed. Failing rather than guessing.")
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"saved pristine copy to {BACKUP.name}")
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {TARGET}")


def restore() -> None:
    if not BACKUP.exists():
        raise SystemExit(f"no pristine copy at {BACKUP}")
    shutil.copy2(BACKUP, TARGET)
    print(f"restored {TARGET}")


def verify() -> None:
    from tiptop.workspace import workspace_cuboids

    cuboids = workspace_cuboids()
    print(f"  {len(cuboids)} workspace obstacles for this rig:")
    for c in cuboids:
        x, y, z = c.pose[:3]
        dx, dy, dz = c.dims
        print(f"    {c.name:28s} centre [{x:+.3f} {y:+.3f} {z:+.3f}] "
              f"dims [{dx:.2f} {dy:.2f} {dz:.2f}]  "
              f"x[{x-dx/2:+.2f},{x+dx/2:+.2f}] y[{y-dy/2:+.2f},{y+dy/2:+.2f}] z[{z-dz/2:+.2f},{z+dz/2:+.2f}]")
    names = {c.name for c in cuboids}
    if any("vention" in n or "nishanth" in n or "ipad" in n for n in names):
        raise SystemExit("still serving MIT LIS geometry")
    print("  MIT LIS geometry is gone")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--restore", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.restore:
        restore()
    elif a.verify:
        verify()
    else:
        install()
        verify()


if __name__ == "__main__":
    main()
