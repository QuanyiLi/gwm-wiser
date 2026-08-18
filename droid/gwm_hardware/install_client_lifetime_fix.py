"""Stop `go_to_q` from tearing down the shared, cached robot client.

Found 2026-08-18 on the first `tiptop-run` of this rig: the run reached the
capture pose, logged "Executed trajectory on the robot", and then died with no
traceback and exit code 1.

`tiptop/motion_planning.py:go_to_q` calls `client.close()` right after
executing its trajectory. `tiptop.utils.get_robot_client` is `@cache`d, so that
`client` is the very same `BambooFrankaClient` object every other caller holds
-- including `tiptop_run`'s `container.robot`. `BambooFrankaClient.close()` is
not a soft close: it does `zmq_context.term()`, and neither
`_send_robotiq_command` nor `_send_panda_hand_command` has any recreate path
(only `_get_latest_state` and the control socket do). So the very next call:

    go_to_capture(...)              # closes the shared client
    container.robot.open_gripper()  # zmq.error.ZMQError: Socket operation on non-socket

The traceback is invisible because `tiptop_run._sync_entrypoint`'s `finally`
runs `sys.exit(exit_code)` with `exit_code` still 1, which replaces the
propagating exception. Reproduced directly:

    a = get_robot_client(); b = get_robot_client()   # a is b -> True
    a.close(); b.open_gripper()                      # ZMQError

This blocks `tiptop-run` outright -- it fails before the instruction prompt, so
nothing downstream (perception, planning, execution) ever gets a chance to run.

The close is harmless for `tiptop/scripts/go_to_conf.py`, a one-shot CLI that
exits immediately afterwards, which is very likely where it came from. In a
long-lived loop it is a lifetime bug: a movement helper must not decide the
lifetime of a process-wide singleton it did not create. `tiptop_run` already
closes the client itself in its own `finally`.

Idempotent, keeps a `.orig`, and `--restore` reverts.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_client_lifetime_fix
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.install_client_lifetime_fix --verify
"""

import argparse
import inspect
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "tiptop/tiptop/motion_planning.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARKER = "# --- patched by gwm_hardware.install_client_lifetime_fix ---"

OLD = """    )
    client.close()
    if not result["success"]:"""
NEW = f"""    )
    {MARKER}
    # Do NOT close here. get_robot_client() is @cache'd, so this is the same
    # object tiptop_run holds as container.robot, and close() terminates the
    # ZMQ context outright -- the next open_gripper() raises ZMQError. The
    # owning process closes it in its own finally.
    if not result["success"]:"""


def install() -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print("already patched")
        return
    if OLD not in text:
        raise SystemExit("tiptop/motion_planning.py does not match what this patch expects "
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


def verify(live: bool) -> None:
    from tiptop.motion_planning import go_to_q

    src = inspect.getsource(go_to_q)
    if "client.close()" in src:
        raise SystemExit("go_to_q still closes the shared client")
    print("  go_to_q no longer closes the shared client")

    if not live:
        print("  (skipping the live check; pass --verify-live to exercise the robot)")
        return

    import numpy as np
    from tiptop.config import tiptop_cfg
    from tiptop.utils import get_robot_client

    client = get_robot_client()
    q0 = np.asarray(client.get_joint_positions())

    # Nudge one joint just past go_to_q's 0.05 rad early-return tolerance, so the
    # plan-execute path -- the one that used to close the client -- actually runs.
    q1 = q0.copy()
    q1[3] += 0.06
    tdf = tiptop_cfg().robot.time_dilation_factor
    print(f"  live: nudging joint 4 by +0.06 rad at time_dilation_factor={tdf}")
    go_to_q(q_target=list(q1), time_dilation_factor=tdf)

    # The exact call that used to raise ZMQError.
    client.open_gripper()
    print("  live: open_gripper() succeeded after a real go_to_q -- client survived")

    go_to_q(q_target=list(q0), time_dilation_factor=tdf)
    print("  live: returned to the starting configuration")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--restore", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--verify-live", action="store_true",
                   help="also move the robot to prove the client survives a real go_to_q")
    a = ap.parse_args()
    if a.restore:
        restore()
    elif a.verify or a.verify_live:
        verify(live=a.verify_live)
    else:
        install()
        verify(live=False)


if __name__ == "__main__":
    main()
