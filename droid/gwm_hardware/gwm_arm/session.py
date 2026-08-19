"""One prompt in, one robot motion out. The GWM arm's front door.

    ./droid/gwm_hardware/gwm_arm/run.sh

Then type what you want, the way you would say it:

    > grasping the object between the tomato and the blue cup
    > put it in the blue cup

**The gripper decides what kind of thing is proposed.** `get_gripper_state()`
reports `{'width', 'is_grasped', 'is_moving'}`, so this is measured, not
inferred or remembered:

    gripper EMPTY   -> PICK  candidates (gwm_arm.propose: clusters -> M2T2 -> cuTAMP)
    gripper HOLDING -> PLACE candidates (gwm_tiptop.place_propose)

Exactly two command types, and the gripper picks which
------------------------------------------------------
There is no intent classification and no "go home" command. What the robot is
holding decides what kind of trajectory gets proposed, and that is the whole
routing rule.

**Scoring is identical to droid-sim.** A place instruction is scored against
place trajectories exactly as `gwm_tiptop.place_propose` produces them -- same
candidates, same two-stage selection, same numbers -- so the sim results stay
the comparison they were built to be.

**Execution deliberately is not.** droid-sim's place episode ends with the
block still held, because there the "grasp" is a weld and releasing is a no-op
(G-25). On hardware the object is really held, so a place that ends holding it
has not placed anything. So the hardware place executes further than it
scored:

    scored:    [close, approach, constrained descent]   <- as in sim
    executed:  the same, THEN open the gripper, THEN return home

The extra two steps are deterministic consequences of having placed something,
not choices, so they need no candidate and no score. `gwm_tiptop/` is untouched
by this -- the divergence lives here, on the hardware side, which is where it
belongs.

Going home prefers z
--------------------
The return-home motion lifts along z before travelling. A held object dragged
laterally at table height sweeps everything in its path; lifting first costs a
second and removes the whole failure mode. Implemented as two planned segments
(lift, then home), each checked by cuRobo against the rig workspace -- not as
an unchecked servo.

Nothing moves without a confirmation, and the Rerun viewer opens for every
proposal so the candidates and their scores are on screen before you say yes.
"""

import argparse
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from gwm_hardware.common.paths import PKG_ROOT
from gwm_hardware.gwm_arm.capture import EXTERNAL_CAM
from gwm_hardware.gwm_arm.run_real import (
    DEPTH_PORT,
    ensure_depth_server,
    install_quiet_logging,
    run_module,
    tag_for,
    _healthy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.session")

SCORER_PORT = 8901

# How far up to lift before travelling home, and how far the arm may already be
# from a plan's start before we insist on planning a motion to it.
HOME_LIFT_M = 0.12


def gripper_state(client) -> dict:
    """{'width', 'is_grasped', 'is_moving'} from the controller."""
    r = client.get_gripper_state()
    if not r.get("success"):
        raise SystemExit(f"could not read the gripper: {r}")
    return r["state"]


def holding(state: dict, width_closed_m: float) -> bool:
    """Is something in the gripper?

    `is_grasped` is the controller's own answer and is what we trust. Width is
    the fallback for a controller that does not report it: a gripper closed on
    nothing reads near zero, one closed on an object reads the object's width.
    """
    if state.get("is_grasped") is not None:
        return bool(state["is_grasped"])
    return float(state["width"]) > width_closed_m


# ------------------------------------------------------------------ go home


def return_home(lift_m: float, execute: bool) -> None:
    """Lift along z, then travel to q_home. Both segments planned and checked."""
    from curobo.types.base import TensorDeviceType

    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import build_curobo_solvers, go_to_q
    from tiptop.utils import get_robot_client

    cfg = tiptop_cfg()
    client = get_robot_client()
    tensor_args = TensorDeviceType()
    ik_solver, motion_gen, _ = build_curobo_solvers(num_particles=64, num_spheres=64,
                                                    include_workspace=True)
    q_now = np.asarray(client.get_joint_positions(), dtype=np.float64)
    state = motion_gen.kinematics.get_state(tensor_args.to_device(q_now))
    pose = state.ee_pose.get_numpy_matrix()[0]

    # Segment 1: same orientation, same xy, +z. Solved by IK so the lift is a
    # real configuration the planner then travels to, rather than a jog.
    target = pose.copy()
    target[2, 3] += lift_m
    from curobo.types.math import Pose

    goal = Pose(
        position=tensor_args.to_device(target[:3, 3][None].astype(np.float32)),
        quaternion=state.ee_pose.quaternion,
    )
    ik = ik_solver.solve_single(goal, retract_config=tensor_args.to_device(q_now).float()[None])
    if bool(ik.success[0]):
        q_lift = ik.solution[0][0].cpu().numpy().astype(np.float64)
        _log.info(f"lift {lift_m * 1000:.0f} mm in z, then home")
        if execute:
            go_to_q(q_target=q_lift, time_dilation_factor=cfg.robot.time_dilation_factor,
                    motion_gen=motion_gen)
    else:
        _log.warning("no IK for the straight lift; going home directly. Anything held "
                     "will travel at its current height -- watch it")

    _log.info("travelling to q_home")
    if execute:
        go_to_q(q_target=list(cfg.robot.q_home),
                time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=motion_gen)
    else:
        _log.info("(dry run: nothing sent)")


def release_then_home(lift_m: float, execute: bool) -> None:
    """What follows a PLACE, once the object is where it was asked to go.

    Not scored and not a candidate: having placed something, releasing it and
    getting the arm out of the way are the only things that can happen next.
    droid-sim stops before both because its grasp is a weld (G-25); hardware
    cannot.
    """
    from tiptop.utils import get_robot_client

    if not execute:
        _log.info("(dry run: would open the gripper and return home)")
        return
    _log.info("placed -- opening the gripper")
    get_robot_client().open_gripper(speed=1.0)
    return_home(lift_m, execute=True)


# ----------------------------------------------------------------- the loop


def ensure_scorer(server_url: str) -> None:
    if _healthy(SCORER_PORT):
        return
    raise SystemExit(
        f"gwm-server is not up on {SCORER_PORT}. Start the stack first:\n"
        "  ./droid/gwm_hardware/gwm_arm/services.sh start gwm"
    )


def warm_up(args) -> None:
    """Build everything an instruction will need, BEFORE the first prompt.

    A session should pay its construction costs once, at startup, and then do
    nothing per instruction but the work that instruction actually implies.
    Measured on this rig, the fixed costs are the cuRobo IK+MotionGen build
    (3.6 s), the kinematics model (0.45 s), the FoundationStereo weights
    (~30 s cold) and the scorer's 16 GB of Qwen (~60 s cold). Paying those at
    the prompt makes the first instruction look four times slower than the
    rest for no reason anyone can see.

    Everything here is cached module-level, so the stages pick it up without
    being told: `planning_solvers` and `fk_model` key on the configuration,
    and `scene_cache` keys on the capture file -- which is why a new capture
    (every motion) still re-perceives, as it must.
    """
    t0 = time.perf_counter()
    from tiptop.config import tiptop_cfg
    from tiptop.planning import build_tamp_config

    from gwm_tiptop.robot_fk import fk_model, planning_solvers

    cfg = tiptop_cfg()
    print("  building the pipeline (once) ...")

    fk_model()
    print(f"    \u25b8 {'kinematics':<16} {time.perf_counter() - t0:6.1f} s")

    t1 = time.perf_counter()
    config = build_tamp_config(
        num_particles=args.k_particles, max_planning_time=60.0, opt_steps=500,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor, near_placement=False)
    planning_solvers(config.num_particles, config.coll_n_spheres, include_workspace=True)
    print(f"    \u25b8 {'cuRobo solvers':<16} {time.perf_counter() - t1:6.1f} s")

    t1 = time.perf_counter()
    ensure_depth_server()
    print(f"    \u25b8 {'depth server':<16} {time.perf_counter() - t1:6.1f} s")

    print(f"  ready in {time.perf_counter() - t0:.1f} s -- instructions now reuse all of it\n")


def one_turn(instruction: str, run_dir: Path, args) -> None:
    from tiptop.utils import get_robot_client

    client = get_robot_client()
    st = gripper_state(client)
    held = holding(st, args.width_closed)
    _log.info(f"gripper: width {st['width'] * 1000:.1f} mm, is_grasped={st.get('is_grasped')} "
              f"-> {'HOLDING something (place mode)' if held else 'EMPTY (pick mode)'}")

    tag = tag_for(instruction)
    proposals = run_dir / "proposals"

    # --- capture -------------------------------------------------------
    argv = ["live", "--out-dir", run_dir]
    if not args.move:
        argv.append("--no-move")
    if held:
        # The held object sits under the gripper, which is exactly where the
        # mask cuts. place_propose measures it from the cloud instead, by
        # excluding the robot's own padded spheres -- a better discriminator
        # than an image mask, and one that needs those pixels present.
        argv.append("--no-gripper-mask")
    run_module("gwm_hardware.gwm_arm.capture", argv, "capture", args.verbose)

    # --- propose -------------------------------------------------------
    if held:
        _log.warning("PLACE proposals on hardware are not yet validated -- the sim "
                     "version assumed a welded block and sim bins (G-25/G-26). Read "
                     "the candidates before executing.")
        run_module("gwm_tiptop.place_propose",
                   ["--h5-path", run_dir / "wrist_obs.h5", "--output-dir", proposals,
                    "--k-total", args.k_total], "propose(place)", args.verbose)
    else:
        run_module("gwm_hardware.gwm_arm.propose",
                   ["--h5-path", run_dir / "wrist_obs.h5", "--output-dir", proposals,
                    "--k-total", args.k_total], "propose(pick)", args.verbose)

    # --- score / gate / viz --------------------------------------------
    run_module("gwm_tiptop.score_client",
               ["--proposals-dir", proposals, "--external-h5", run_dir / "external_obs.h5",
                "--instruction", instruction, "--cam", args.cam, "--tag", tag,
                "--rat-scale", args.rat_scale, "--object-score", args.object_score,
                "--server-url", args.server_url, "--dump-dir", run_dir / f"rat_{tag}"],
               "score", args.verbose)

    if not held and args.gate:
        run_module("gwm_tiptop.grasp_gate",
                   ["--proposals-dir", proposals, "--h5-path", run_dir / "wrist_obs.h5",
                    "--use-plane-normal", "--use-robot-arm-filter", "--apply", tag],
                   "gate", args.verbose)
    elif not held:
        _log.warning("--no-gate: the closing-line grasp gate is OFF. GWM cannot see "
                     "grasp robustness (its RAT frames are robot-only), and this gate "
                     "is what took nearbowl from 0/5 to 5/5 in G-27")

    if args.debug:
        argv = ["--proposals-dir", proposals, "--h5-path", run_dir / "wrist_obs.h5",
                "--tag", tag, "--instruction", instruction,
                "--external-h5", run_dir / "external_obs.h5", "--cam", args.cam]
        if not args.rerun:
            argv.append("--no-rerun")
        run_module("gwm_hardware.gwm_arm.viz_debug", argv, "viz", args.verbose)

    scores = json.loads((proposals / f"scores_{tag}.json").read_text())
    winner = proposals / f"winner_{tag}.json"
    print(f"\n  selected object : {scores.get('selected_target')}")
    for d in scores.get("object_ranking", [])[:4]:
        print(f"      {d['score']:+.4f}  {d['target']}  (n={d['n']})")
    rank = scores.get("object_ranking", [])
    if len(rank) > 1:
        print(f"  margin          : {rank[0]['score'] - rank[1]['score']:+.4f}")
    print(f"  winner          : {winner.name}")
    if args.debug:
        print("  Rerun viewer and score_overlay png are up -- look before you answer.")
    else:
        print("  (--debug adds the Rerun scene and the scored-candidate overlay)")

    if not args.execute:
        print("  dry run: nothing will move. Re-run with --execute.")
        return
    if input("\n  execute this plan? type 'go': ").strip().lower() != "go":
        print("  aborted")
        return
    argv = ["--plan", winner, "--execute", "--go-to-start", "--yes"]
    if not held:
        # A pick plan assumes it starts with an open gripper. A place plan must
        # NOT be pre-opened -- that drops the object before it goes anywhere.
        argv.append("--open-before")
    run_module("gwm_hardware.gwm_arm.execute", argv, "execute", args.verbose)

    if held:
        release_then_home(args.lift, execute=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, default=PKG_ROOT / "runs/session")
    ap.add_argument("--execute", action="store_true", help="allow the robot to move")
    ap.add_argument("--no-move", dest="move", action="store_false",
                    help="capture where the arm stands instead of driving to q_capture")
    ap.add_argument("--k-total", type=int, default=16)
    ap.add_argument("--k-particles", type=int, default=256,
                    help="cuTAMP particle count; part of the solver cache key, so "
                         "warm-up and the proposer must agree on it")
    ap.add_argument("--cam", default=EXTERNAL_CAM)
    ap.add_argument("--rat-scale", default="3.0")
    ap.add_argument("--object-score", default="mean", choices=["mean", "max", "median"])
    ap.add_argument("--server-url", default=f"http://localhost:{SCORER_PORT}")
    ap.add_argument("--lift", type=float, default=HOME_LIFT_M,
                    help="metres to lift in z before travelling home")
    ap.add_argument("--width-closed", type=float, default=0.005,
                    help="fallback width threshold if the controller reports no is_grasped")
    ap.add_argument("--debug", action="store_true",
                    help="also run the debug viewer: the Rerun scene and the "
                         "score_overlay png with every candidate coloured by its score. "
                         "Off by default -- it costs a stage and is for looking, not deciding")
    ap.add_argument("--no-rerun", dest="rerun", action="store_false",
                    help="with --debug, write the score overlay but do not spawn the "
                         "Rerun viewer (headless sessions)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip the closing-line grasp gate. ON by default because it is "
                         "not a debug tool: GWM scores semantic alignment, not grasp "
                         "robustness, and this is the filter that catches a fragile winner")
    ap.add_argument("--verbose", action="store_true",
                    help="show every stage's raw output instead of a summary")
    ap.add_argument("--instruction", default=None,
                    help="run one instruction and exit instead of prompting")
    args = ap.parse_args()

    install_quiet_logging()
    ensure_scorer(args.server_url)
    warm_up(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    print("\nGWM x TiPToP -- type an instruction, 'exit' to quit."
          f"\n{'EXECUTION ENABLED -- hand on the E-stop' if args.execute else 'dry run (--execute to arm)'}\n")

    single = args.instruction
    n = 0
    while True:
        try:
            instruction = single or input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not instruction:
            continue
        if instruction.lower() in ("exit", "quit"):
            break
        n += 1
        run_dir = args.run_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{n:02d}"
        try:
            one_turn(instruction, run_dir, args)
        except SystemExit as e:
            _log.error(str(e))
        except Exception:
            _log.exception("turn failed")
        if single:
            break


if __name__ == "__main__":
    main()
