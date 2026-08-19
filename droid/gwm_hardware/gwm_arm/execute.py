"""Execute a selected plan on the robot.

`tiptop.execute_plan.execute_cutamp_plan` takes cuTAMP's in-memory plan --
`step["plan"]` is a live cuRobo tensor object. The GWM arm's winner arrives as
a `serialize_plan` JSON file, hours or a machine apart from the planner that
produced it, so it needs an executor of its own. The controller calls are
identical, deliberately: same `execute_joint_impedance_path`, same
open/close, same order, so an A/B comparison is not confounded by how the two
arms drive the robot.

What is added is the checking that an offline plan needs and an in-memory one
does not:

  * **The plan is only valid from where it was planned.** Every waypoint is
    absolute, and the first one is the capture pose the observation was taken
    at. Executing it from somewhere else jumps the arm to that first waypoint
    through whatever is in between. So the current configuration is compared
    against the plan's `q_init` and execution refuses on a mismatch;
    `--go-to-start` plans a checked cuRobo motion there first.
  * **Nothing moves without being asked twice.** `--execute` is required, and
    an interactive session confirms at the prompt on top of that. The default
    prints the plan and stops.
  * **The gripper stays shut at the end.** A pick plan ends holding the
    object; opening drops it. tiptop asks before opening and so does this.

    python -m gwm_hardware.gwm_arm.execute \
        --plan runs/gwm/scene01/proposals/winner_pick_cup.json          # dry run
    python -m gwm_hardware.gwm_arm.execute --plan ... --execute --go-to-start
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.execute")

# Per-joint tolerance between where the arm is and where the plan starts.
# 0.02 rad is ~1.1 deg; the deterministic go-to-capture repeats far tighter
# than that, so anything larger means the arm is somewhere else entirely.
Q_INIT_TOL_RAD = 0.02
# A trajectory step must START where the arm already is. The controller does
# not interpolate into a path: it is handed waypoints and follows them, so a
# first waypoint far from the current configuration is a commanded jump.
#
# On 2026-08-19 a place plan was handed one 0.885 rad (51 deg) from the
# capture pose and the controller aborted with "Trajectory execution failed"
# -- a message that says nothing about the cause, and the same message a
# collision or a limit violation produces. The arm did not move (drift
# 0.0004 rad), so the controller's own guard held; this one exists so the
# failure is named BEFORE anything is sent, and so a controller with a laxer
# guard cannot be handed the jump at all.
MAX_STEP_JUMP_RAD = 0.05


def check_continuity(plan: dict, q_now) -> None:
    """Every trajectory must begin where the previous step left the arm.

    Checked over the whole plan before a single waypoint is sent, because the
    interesting failure is a plan that executes its first step and faults on
    its second, leaving the arm somewhere no later step expects.
    """
    import numpy as _np

    q = _np.asarray(q_now, dtype=_np.float64)
    for i, step in enumerate(plan["steps"]):
        if step["type"] != "trajectory":
            continue
        pos = _np.asarray(step["positions"], dtype=_np.float64)
        gap = float(_np.abs(pos[0] - q).max())
        if gap > MAX_STEP_JUMP_RAD:
            j = int(_np.argmax(_np.abs(pos[0] - q)))
            raise ValueError(
                f"step {i + 1} ({step.get('label', 'trajectory')}) starts {gap:.4f} rad "
                f"({_np.degrees(gap):.1f} deg) from where the arm will be, worst on joint "
                f"{j + 1} ({q[j]:+.4f} -> {pos[0][j]:+.4f}). Limit {MAX_STEP_JUMP_RAD} rad. "
                f"This plan commands a jump; refusing to send it."
            )
        q = pos[-1]


def describe(plan: dict) -> str:
    out, t = [], 0.0
    for i, step in enumerate(plan["steps"]):
        if step["type"] == "trajectory":
            n = len(step["positions"])
            dur = n * step["dt"]
            t += dur
            out.append(f"  {i + 1}. trajectory  {n:4d} waypoints  {dur:5.2f} s  "
                       f"{step.get('label', '')}")
        else:
            out.append(f"  {i + 1}. gripper     {step['action']:<5}  "
                       f"{step.get('label', '')}")
    out.append(f"  total motion time {t:.2f} s (before time_dilation_factor)")
    return "\n".join(out)


def execute_serialized(plan: dict, client) -> None:
    """The controller calls of execute_cutamp_plan, over the serialized schema."""
    import time

    from tiptop.execute_plan import ExecutionFailure

    check_continuity(plan, client.get_joint_positions())

    start = time.perf_counter()
    for i, step in enumerate(plan["steps"]):
        label = step.get("label", "")
        if step["type"] == "gripper":
            _log.info(f"step {i + 1}/{len(plan['steps'])}: gripper {step['action']} ({label})")
            result = (client.open_gripper(speed=1.0) if step["action"] == "open"
                      else client.close_gripper(speed=1.0))
        elif step["type"] == "trajectory":
            pos = np.asarray(step["positions"], dtype=np.float64)
            vel = np.asarray(step["velocities"], dtype=np.float64)
            _log.info(f"step {i + 1}/{len(plan['steps'])}: trajectory, "
                      f"{len(pos)} waypoints ({label})")
            result = client.execute_joint_impedance_path(
                joint_confs=pos, joint_vels=vel, durations=[step["dt"]] * len(pos))
        else:
            raise ValueError(f"unknown step type {step['type']!r}")
        if result is None:
            raise RuntimeError("controller returned None")
        if not result["success"]:
            raise ExecutionFailure(result["error"])
    _log.info(f"execution finished in {time.perf_counter() - start:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path, help="a winner_TAG.json / plan_NN.json")
    ap.add_argument("--execute", action="store_true", help="actually drive the robot")
    ap.add_argument("--go-to-start", action="store_true",
                    help="plan a cuRobo motion to the plan's q_init first, if the arm has drifted")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    ap.add_argument("--open-before", action="store_true",
                    help="open the gripper before running the plan. Correct for a PICK, "
                         "which assumes it starts open; WRONG for a place, which starts "
                         "holding the object")
    ap.add_argument("--open-after", action="store_true",
                    help="open the gripper when the plan finishes (drops whatever is held)")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    q_init = np.asarray(plan["q_init"], dtype=np.float64)
    print(f"\n{args.plan}")
    print(f"  q_init {np.round(q_init, 4).tolist()}")
    print(describe(plan))

    if not args.execute:
        print("\ndry run: nothing was sent to the robot. Add --execute to run it.")
        return

    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import go_to_q
    from tiptop.utils import get_robot_client

    cfg = tiptop_cfg()
    client = get_robot_client()
    q_now = np.asarray(client.get_joint_positions(), dtype=np.float64)
    drift = np.abs(q_now - q_init)
    _log.info(f"arm is at {np.round(q_now, 4).tolist()}; worst drift from the plan's "
              f"start {drift.max():.4f} rad on joint {int(drift.argmax()) + 1}")

    if drift.max() > Q_INIT_TOL_RAD:
        if not args.go_to_start:
            raise SystemExit(
                f"the arm is {drift.max():.3f} rad from where this plan was planned "
                f"(tolerance {Q_INIT_TOL_RAD}). Every waypoint is absolute, so running "
                "it from here would jump the arm to the first waypoint through "
                "whatever is in the way. Re-run with --go-to-start to plan a checked "
                "motion there first, or re-capture and re-plan from where the arm is."
            )
        _log.info("planning a motion to the plan's start configuration")
        go_to_q(q_target=q_init, time_dilation_factor=cfg.robot.time_dilation_factor)
        q_now = np.asarray(client.get_joint_positions(), dtype=np.float64)
        if np.abs(q_now - q_init).max() > Q_INIT_TOL_RAD:
            raise SystemExit(f"still {np.abs(q_now - q_init).max():.3f} rad away after "
                             "the move; not executing")

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("no tty to confirm on; pass --yes if you really mean it")
        print(f"\nAbout to drive the robot at time_dilation_factor "
              f"{cfg.robot.time_dilation_factor}. Hand on the E-stop.")
        if input("type 'go' to execute: ").strip().lower() != "go":
            raise SystemExit("aborted")

    if args.open_before:
        # A PICK plan assumes it starts with an open gripper. A PLACE plan must
        # never be pre-opened: it starts holding the object, and opening first
        # drops it before it goes anywhere. Off by default so the dangerous
        # case is the one you have to ask for.
        _log.info("opening the gripper before the plan (--open-before)")
        client.open_gripper()
    execute_serialized(plan, client)

    if args.open_after:
        print("WARNING: the object will drop when the gripper opens. Be ready to catch it.")
        if args.yes or input("open gripper? [y]: ").strip().lower() == "y":
            client.open_gripper()
    else:
        _log.info("leaving the gripper closed (pass --open-after to release)")


if __name__ == "__main__":
    main()
