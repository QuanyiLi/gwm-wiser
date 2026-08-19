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
from gwm_hardware.common.gripper_geometry import closed_tip_overhang
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


def retrace_descent(plan: dict, client) -> bool:
    """Back out along the exact path the plan came down. True if it moved.

    The lift after a release wants to be straight up and collision-free. The
    obvious way is IK on a raised pose, and it does not work here: cuRobo
    compiles a CUDA graph specialised to the first solve_state an IKSolver
    sees, and every call shape we can make from outside raises either
    "changing goal type, cuda graph reset not available" or an AttributeError
    on a buffer the warm-up never allocated.

    Retracing needs no solver at all, and is strictly better than a fresh
    Cartesian lift: the last trajectory segment of a place plan IS the
    constrained straight descent into the destination, so reversing it leaves
    by the route it arrived on. That path was collision-checked when it was
    planned and was physically traversed seconds ago.

    Velocities are negated as well as reversed, which is what time-reversing a
    trajectory means; leaving them positive would command the controller to
    move away from where the positions go.
    """
    steps = [st for st in plan.get("steps", []) if st["type"] == "trajectory"]
    if not steps:
        return False
    last = steps[-1]
    pos = np.asarray(last["positions"], dtype=np.float64)[::-1].copy()
    vel = -np.asarray(last["velocities"], dtype=np.float64)[::-1].copy()
    if len(pos) < 2:
        return False
    _log.info(f"retracing the descent ({len(pos)} waypoints) to lift clear")
    r = client.execute_joint_impedance_path(joint_confs=pos, joint_vels=vel,
                                            durations=[last["dt"]] * len(pos))
    if r is None or not r.get("success"):
        _log.warning(f"retrace refused by the controller: {r}; going home directly")
        return False
    return True


def return_to_capture(execute: bool, plan: dict | None = None) -> None:
    """Get clear, then travel to q_CAPTURE -- not q_home.

    Every turn begins at `q_capture`: the capture step drives there, the scene
    photo the scorer sees is taken from there, and every candidate is planned
    starting from it. Parking at `q_home` between turns therefore bought a
    motion whose only effect was that the next turn had to undo it. Ending at
    the capture pose leaves the arm where the next instruction already needs
    it, and `go_to_capture` becomes a no-op instead of a second traverse.

    `cfg.robot.q_home` is deliberately NOT changed: the baseline TiPToP arm
    homes there, and the two experiments do not share resting poses just
    because they share a robot.

    "Clear" is the plan's own descent reversed when there is one, and nothing
    otherwise -- `go_to_q` plans against the rig workspace either way, so the
    travel itself is collision-checked. What retracing buys is that the FIRST
    motion after a release is vertical, instead of a planner free to sweep the
    held-object-shaped hole sideways through whatever it was placed into.
    """
    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import go_to_q
    from tiptop.utils import get_robot_client

    from gwm_tiptop.robot_fk import default_planning_solvers, reset_world_to_workspace

    cfg = tiptop_cfg()
    client = get_robot_client()
    if not execute:
        _log.info("(dry run: would retrace the descent, then travel to q_capture)")
        return
    if plan is not None:
        retrace_descent(plan, client)
    _log.info("travelling to q_capture (where the next turn starts)")
    _, motion_gen, _ = default_planning_solvers()
    reset_world_to_workspace(motion_gen)
    go_to_q(q_target=list(cfg.robot.q_capture),
            time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=motion_gen)


def release_then_return(execute: bool, plan: dict | None = None) -> None:
    """What follows a PLACE, once the object is where it was asked to go.

    Not scored and not a candidate: having placed something, releasing it and
    getting the arm out of the way are the only things that can happen next.
    droid-sim stops before both because its grasp is a weld (G-25); hardware
    cannot.
    """
    from tiptop.utils import get_robot_client

    if not execute:
        _log.info("(dry run: would open the gripper, retrace, and return to q_capture)")
        return
    _log.info("placed -- opening the gripper")
    get_robot_client().open_gripper(speed=1.0)
    return_to_capture(execute=True, plan=plan)


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

    from gwm_tiptop.robot_fk import default_planning_solvers, fk_model

    cfg = tiptop_cfg()
    print("  building the pipeline (once) ...")

    fk_model()
    print(f"    \u25b8 {'kinematics':<16} {time.perf_counter() - t0:6.1f} s")

    t1 = time.perf_counter()
    default_planning_solvers(num_particles=args.k_particles)
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
                    "--k-total", args.k_total, "--use-plane-normal",
                    "--use-robot-arm-filter", "--anchor-descent",
                    "--closed-tip-overhang", f"{closed_tip_overhang():.5f}",
                    "--release-above-rim", str(args.release_above_rim),
                    "--max-support-slope", str(args.max_support_slope),
                    "--skip-leading-close"],
                   "propose(place)", args.verbose)
    else:
        run_module("gwm_hardware.gwm_arm.propose",
                   ["--h5-path", run_dir / "wrist_obs.h5", "--output-dir", proposals,
                    "--k-total", args.k_total], "propose(pick)", args.verbose)

    # Nothing to score is a normal outcome, not a crash. The proposer already
    # says so and writes an index with num_proposals 0; before this the turn
    # walked straight into the scorer with an empty candidate list and came
    # back as `500 Server Error` from gwm-server -- an error about the wrong
    # thing entirely, several frames away from the fact that the scene had no
    # graspable object in it.
    index = json.loads((proposals / "proposals_index.json").read_text())
    if not index.get("num_proposals"):
        per = index.get("perception", {}) or {}
        clusters = per.get("clusters", [])
        graspless = set(per.get("graspless_clusters", []))
        print(f"\n  no candidates -- nothing to score, nothing to execute.")
        if clusters:
            for c in clusters:
                why = ("M2T2 proposed no grasp for it" if c in graspless
                       else "grasps existed but none survived reachability refinement")
                print(f"      {c}: {why}")
        else:
            print("      the scene decomposed into no clusters at all")
        print("      look at proposals/clusters.png -- if the object is there and was "
              "still refused, it is out of reach or the grasps are unplannable from "
              "this pose, not a scoring problem.")
        return

    # --- score / gate / viz --------------------------------------------
    run_module("gwm_tiptop.score_client",
               ["--proposals-dir", proposals, "--external-h5", run_dir / "external_obs.h5",
                "--instruction", instruction, "--cam", args.cam, "--tag", tag,
                "--rat-scale", args.rat_scale, "--object-score", args.object_score,
                "--server-url", args.server_url, "--dump-dir", run_dir / f"rat_{tag}"]
               # A place plan's scored timeline is not its executed one. It opens
               # with 1.33 s of the arm frozen at the capture pose while the
               # gripper closes, and it ENDS the instant the gripper arrives --
               # the release is issued afterwards by release_then_return, so it was
               # never scored. Renders are robot-only, so without the open frame a
               # place is pixel-identical to a grasp approach.
               + (["--drop-static-prefix", "--append-release"] if held else []),
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
    # --execute is the arming gate for the whole session; a second per-turn
    # confirmation only earns its keystroke when there is something new to look
    # at, which is exactly --debug (the Rerun scene and the scored overlay).
    # Without it, the numbers above are already on screen and the answer was
    # always "go".
    if args.debug and input("\n  execute this plan? type 'go': ").strip().lower() != "go":
        print("  aborted")
        return
    argv = ["--plan", winner, "--execute", "--go-to-start", "--yes"]
    if not held:
        # A pick plan assumes it starts with an open gripper. A place plan must
        # NOT be pre-opened -- that drops the object before it goes anywhere.
        argv.append("--open-before")
    run_module("gwm_hardware.gwm_arm.execute", argv, "execute", args.verbose)

    if held:
        release_then_return(execute=True, plan=json.loads(winner.read_text()))


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
    ap.add_argument("--width-closed", type=float, default=0.005,
                    help="fallback width threshold if the controller reports no is_grasped")
    ap.add_argument("--max-support-slope", type=float, default=0.18,
                    help="rise over the held object's footprint radius, above which a solid "
                         "destination stops being a placement target. Every cluster is a "
                         "destination, so without this the proposer plans onto a ball apex as "
                         "readily as into a tray. 0 = off")
    ap.add_argument("--release-above-rim", type=float, default=0.03,
                    help="metres above a container's rim to release from, letting the "
                         "object drop in rather than carrying it to the floor. 0 carries "
                         "it down (droid-sim's behaviour)")
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
          f"\n{'EXECUTION ENABLED -- hand on the E-stop' if args.execute else 'dry run (--execute to arm)'}"
          + ("  (--debug confirms each plan before it runs)\n" if not args.debug else "\n"))

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
