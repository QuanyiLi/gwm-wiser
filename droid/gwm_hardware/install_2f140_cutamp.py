"""Make `robot.type: panda_robotiq` mean **2F-140** on this rig.

`tiptop/motion_planning.py` dispatches on `cfg.robot.type` straight into
`cutamp.robots.franka_robotiq`, whose `panda_robotiq_*` functions hard-code
`panda_robotiq_2f_85.yml` and `get_robotiq_2f_85_gripper_spheres`. So the
baseline `tiptop-run` arm plans with a gripper **62 mm too short** unless
cuTAMP is redirected -- and the baseline has to be correct too, or the A/B is
comparing against an arm that is quietly unsafe.

Our own drivers can just import `gwm_hardware.robot_2f140`; `tiptop-run`
cannot, hence this. It rewrites three literals in cuTAMP's
`robots/franka_robotiq.py`:

    assets_dir      -> gwm_hardware/assets
    2f_85.yml       -> panda_robotiq_2f_140.yml
    2F-85 spheres   -> gwm_hardware.robot_2f140.get_gripper_spheres()

The cuTAMP clone is gitignored and rebuilt by `install-cutamp.sh`, so this is
a **replayable install step, not a fork** (G-4 stays intact: the algorithm is
untouched, only which robot it loads). Re-run it after any cuTAMP reinstall.
`--restore` puts the original file back from the `.orig` copy it keeps.

Only the `panda_robotiq_*` entry points are touched; `fr3_robotiq_*` and the
UR5 are left alone.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_2f140_cutamp
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_2f140_cutamp --verify
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_2f140_cutamp --restore
"""

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "tiptop/cutamp/cutamp/robots/franka_robotiq.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARKER = "# --- patched by gwm_hardware.install_2f140_cutamp ---"

PATCHES = [
    # 1. load our config instead of the 2F-85 one
    (
        """def panda_robotiq_curobo_cfg():
    assets_dir = Path(__file__).parent / "assets"
    cfg = load_yaml(str(assets_dir / "panda_robotiq_2f_85.yml"))""",
        f"""def panda_robotiq_curobo_cfg():
    {MARKER}
    # The zhiwei rig carries a Robotiq 2F-140, not the 2F-85 cuTAMP ships; its
    # TCP sits 62 mm further out, so the stock model would drive the real
    # fingers that much too deep on every grasp.
    #
    # This redirect is GLOBAL to `panda_robotiq`, and droid-sim-evals reads the
    # same robot type (`tiptop_websocket_server.py:81`) while its simulated
    # robot IS a 2F-85 (`franka_robotiq_2f_85_flattened.usd`). A sim run on a
    # patched checkout would therefore plan the wrong gripper. Escape hatch:
    #
    #     GWM_TIPTOP_GRIPPER=2f85   -> upstream behaviour, use this for droid-sim
    #
    # 2F-140 is the default because the two failure directions are not
    # symmetric: a sim run with the wrong gripper yields bad numbers, a
    # HARDWARE run with the wrong gripper drives 62 mm into the table.
    import logging as _lg
    import os as _os
    from pathlib import Path as _Path

    _use_85 = _os.environ.get("GWM_TIPTOP_GRIPPER") == "2f85"
    _lg.getLogger(__name__).info(
        "panda_robotiq -> %s", "2F-85 (upstream)" if _use_85 else "2F-140 (zhiwei rig)")
    if _use_85:
        assets_dir = _Path(__file__).parent / "assets"
        cfg = load_yaml(str(assets_dir / "panda_robotiq_2f_85.yml"))
    else:
        assets_dir = _Path("{HERE}") / "assets"
        cfg = load_yaml(str(assets_dir / "panda_robotiq_2f_140.yml"))""",
    ),
    # 2. gripper spheres for cuTAMP's grasp filter
    (
        """    Get the collision spheres for the Robotiq 2F-85 gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    \"\"\"
    return get_robotiq_2f_85_gripper_spheres(tensor_args)""",
        f"""    Get the collision spheres for this rig's Robotiq 2F-140 gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    {MARKER}  GWM_TIPTOP_GRIPPER=2f85 restores upstream.
    \"\"\"
    import os as _os

    if _os.environ.get("GWM_TIPTOP_GRIPPER") == "2f85":
        return get_robotiq_2f_85_gripper_spheres(tensor_args)
    from gwm_hardware.robot_2f140 import get_gripper_spheres as _spheres_2f140

    return _spheres_2f140(tensor_args)""",
    ),
]


def _panda_section(text: str) -> str:
    """The file defines fr3_* first, then panda_*; only patch the panda half."""
    return text[text.index("panda_robotiq_neutral_joint_positions"):]


def install() -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print("already patched")
        return
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"saved pristine copy to {BACKUP.name}")

    head_len = len(text) - len(_panda_section(text))
    head, tail = text[:head_len], text[head_len:]
    for old, new in PATCHES:
        if old not in tail:
            raise SystemExit(
                "cuTAMP's franka_robotiq.py does not match what this patch expects "
                f"-- upstream changed. Failing rather than guessing.\nMissing:\n{old[:120]}...")
        tail = tail.replace(old, new, 1)
    TARGET.write_text(head + tail)
    print(f"patched {TARGET}")


def restore() -> None:
    if not BACKUP.exists():
        raise SystemExit(f"no pristine copy at {BACKUP}; reinstall cuTAMP instead")
    shutil.copy2(BACKUP, TARGET)
    print(f"restored {TARGET}")


def verify() -> None:
    """Load through cuTAMP's own entry points and confirm they yield the 2F-140."""
    import numpy as np
    from cutamp.robots.franka_robotiq import (
        get_panda_robotiq_gripper_spheres,
        get_panda_robotiq_kinematics_model,
        panda_robotiq_curobo_cfg,
    )

    cfg = panda_robotiq_curobo_cfg()["robot_cfg"]["kinematics"]
    urdf = cfg["urdf_path"]
    ok_urdf = "2f_140" in urdf
    print(f"  urdf_path             : {urdf}  {'OK' if ok_urdf else 'STILL 2F-85'}")

    model = get_panda_robotiq_kinematics_model()
    import torch
    from curobo.types.base import TensorDeviceType
    q = torch.zeros((1, len(model.joint_names)), device=TensorDeviceType().device)
    q[0, :7] = torch.tensor([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0])
    ee = model.get_state(q).ee_position[0].cpu().numpy()
    print(f"  ee ({cfg['ee_link']}) at neutral: {np.round(ee, 4).tolist()}")

    spheres = get_panda_robotiq_gripper_spheres()
    reach = float(spheres[:, 2].max())
    ok_reach = abs(reach - 0.2145) < 5e-3
    print(f"  gripper spheres       : {tuple(spheres.shape)}, z reach {reach * 1000:.1f} mm "
          f"{'OK (2F-140)' if ok_reach else 'NOT the 2F-140 set'}")

    if not (ok_urdf and ok_reach):
        raise SystemExit("verification FAILED -- cuTAMP is still serving the 2F-85")
    print("  cuTAMP now serves the 2F-140 through its panda_robotiq entry points")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--restore", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.restore:
        restore()
    elif args.verify:
        verify()
    else:
        install()
        verify()


if __name__ == "__main__":
    main()
