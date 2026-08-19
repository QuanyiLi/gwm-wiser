"""One instruction, end to end, on the real robot.

    capture -> propose -> score -> gate -> viz -> execute

Each stage is a separate process, and that is the design, not laziness. The
planner stack (cuRobo + cuTAMP + M2T2 client, 6-10 GB) and the GWM scorer
(~20 GB resident) were kept off the same GPU at the same time on the 3090
(G-8); this rig's 5090 has 32 GB and could probably hold both, but "probably"
is not a thing to discover halfway through a hardware run. Process boundaries
give the sequencing for free -- CUDA memory goes back when the process exits --
and they also mean any stage can be re-run on its own against the artefacts
the previous one left on disk, which is what actually happens while debugging.

Everything runs in the tiptop pixi env. The one component that does not is
`gwm-server`, which owns the pinned `transformers==4.57.6` environment; it is a
long-lived service started separately (`gwm_arm/services.sh`) and reached over
HTTP, so this driver never has to load it.

    # dry run on a scene, no robot motion at all: score and look
    python -m gwm_hardware.gwm_arm.run_real \
        --run-dir runs/gwm/scene01 --instruction "pick up the blue cup"

    # the same, driving the robot
    python -m gwm_hardware.gwm_arm.run_real \
        --run-dir runs/gwm/scene01 --instruction "pick up the blue cup" --execute

    # re-score a captured scene with a different instruction (seconds, no robot)
    python -m gwm_hardware.gwm_arm.run_real --run-dir runs/gwm/scene01 \
        --instruction "pick up the red box" --stages score,gate,viz
"""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from gwm_hardware.common.paths import REPO_ROOT
from gwm_hardware.gwm_arm.capture import EXTERNAL_CAM, EXTERNAL_CAM_2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.run_real")

STAGES = ["capture", "propose", "score", "gate", "viz", "execute"]
PIXI = ["pixi", "run", "--manifest-path", "droid/tiptop/pixi.toml", "python"]
DEPTH_PORT = 1234       # FoundationStereo


def _healthy(port: int) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_depth_server() -> None:
    """Bring FoundationStereo up if it is down -- `capture` cannot work without it.

    It is started and stopped around the capture stage rather than left running,
    because it is the single largest resident consumer on this card and it is
    idle for every stage but one.
    """
    if _healthy(DEPTH_PORT):
        return
    _log.info("FoundationStereo is down; starting it (weights take ~30 s)")
    subprocess.Popen(
        "nohup pixi run server > "
        f"{REPO_ROOT}/droid/gwm_hardware/.service-logs/fs.log 2>&1",
        shell=True, cwd=REPO_ROOT / "droid/FoundationStereo",
        start_new_session=True)
    for _ in range(40):
        if _healthy(DEPTH_PORT):
            _log.info("FoundationStereo up")
            return
        time.sleep(3)
    raise SystemExit(
        f"FoundationStereo did not come up on {DEPTH_PORT}; see "
        "droid/gwm_hardware/.service-logs/fs.log")


def tag_for(instruction: str) -> str:
    """A filesystem-safe, COLLISION-FREE tag for one instruction.

    The readable part is truncated, so it alone is not unique -- "...between the
    blue cup and the tomato" and "...between the blue cup and the Oreo box" cut
    to the same 48 characters, and the second scoring silently overwrote the
    first's scores and winner (found 2026-08-19, the hard way). A short digest
    of the FULL instruction is appended so two different instructions can never
    land on the same artefacts.
    """
    t = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")
    digest = hashlib.sha1(instruction.encode()).hexdigest()[:6]
    return f"{t[:40] or 'gwm'}_{digest}"


def run(cmd: list[str], what: str) -> None:
    _log.info(f"[{what}] {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[{what}] failed with exit code {r.returncode}")
    _log.info(f"[{what}] done in {time.perf_counter() - t0:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k-total", type=int, default=16)
    # ONE scoring view by default, deliberately, while the rig is being brought
    # up: one camera means one extrinsic to calibrate, one overlay gate to
    # read, and one thing to blame when a score looks wrong. `external_cam` is
    # the side-view D435 -- it already frames the arm, the gripper and the
    # whole tabletop, and it shoots against the black backdrop rather than into
    # the window. The head-on D435i (`external_cam_2`) sees the arm better but
    # is aimed straight at a window, and RGB is the ONLY thing the scorer gets.
    #
    # The upgrade, once both views are calibrated and both pass the overlay
    # gate, is one flag: `--cam external_cam,external_cam_2`. score_client
    # then averages each candidate's score across the views before the
    # unchanged two-stage selection. On the sim that beat every single view and
    # every other fusion rule under score noise, because the views genuinely
    # disagree (r=+0.58 between their per-object score deviations, and a
    # between-view spread as large as the semantic signal itself -- G-30).
    ap.add_argument("--cam", default=EXTERNAL_CAM,
                    help="scoring viewpoint(s), comma-separated for fusion. "
                         f"Known: {EXTERNAL_CAM}, {EXTERNAL_CAM_2}")
    ap.add_argument("--rat-scale", default="3.0")
    ap.add_argument("--object-score", default="mean", choices=["mean", "max", "median"])
    ap.add_argument("--server-url", default="http://localhost:8901")
    ap.add_argument("--replay-run", type=Path,
                    help="capture by replaying a saved tiptop_outputs/eval run "
                         "instead of reading the cameras")
    ap.add_argument("--no-move", dest="move", action="store_false",
                    help="capture without driving to q_capture")
    ap.add_argument("--execute", action="store_true", help="let the execute stage move the robot")
    ap.add_argument("--go-to-start", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--horizontal-cut", dest="use_plane_normal", action="store_false",
                    help="droid-sim's world-z above-table cut (see gwm_arm.propose)")
    ap.add_argument("--gate-min-slab-pts", type=int, default=None,
                    help="rig-dependent point-count threshold; see magic_numbers.md #8")
    ap.add_argument("--no-rerun", dest="rerun", action="store_false",
                    help="skip the Rerun viewer in the viz stage (headless sessions)")
    ap.add_argument("--free-depth", action="store_true",
                    help="tear FoundationStereo down after capture. Only needed on a "
                         "card too small to hold every module at once; since the "
                         "memory-release patch it costs 2.0 GB, so the default keeps it")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}; known: {STAGES}")

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    proposals = run_dir / "proposals"
    wrist_h5 = run_dir / "wrist_obs.h5"
    external_h5 = run_dir / "external_obs.h5"
    tag = args.tag or tag_for(args.instruction)
    _log.info(f"run_dir {run_dir}  tag {tag}  stages {stages}")

    if "capture" in stages and not args.replay_run:
        ensure_depth_server()

    if "capture" in stages:
        if args.replay_run:
            run(PIXI + ["-m", "gwm_hardware.gwm_arm.capture", "replay",
                        args.replay_run, "--out-dir", run_dir], "capture")
        else:
            cmd = PIXI + ["-m", "gwm_hardware.gwm_arm.capture", "live", "--out-dir", run_dir]
            if not args.move:
                cmd.append("--no-move")
            run(cmd, "capture")

    # VRAM budget on this 32 GB card, measured 2026-08-19:
    #
    #   gwm-server       19100 MiB   score
    #   FoundationStereo  2072 MiB   capture        (was 9342 -- see below)
    #   M2T2              1180 MiB   propose
    #   cuRobo             740 MiB   propose / gate / viz
    #   ------------------------------------------------------------
    #                    23092 MiB of 32607 -> ~9.5 GB spare, ALL co-resident
    #
    # The first live run OOM'd here because FoundationStereo sat at 9342 MiB,
    # of which 5684 MiB was PyTorch allocator cache retained after a single
    # 1280x720 forward. `common/install_fs_memory_release.py` releases it; the
    # server now settles at 2072 MiB with identical depth (96.7 % valid) and no
    # measured slowdown. That is what makes every module online at once, rather
    # than the modules taking turns.
    #
    # Nothing needs tearing down mid-run any more. `--free-depth` keeps the
    # old behaviour for a smaller card.
    if "capture" in stages and not args.replay_run and args.free_depth \
            and any(s in stages for s in ("propose", "score", "gate", "viz")):
        _log.info("--free-depth: tearing down FoundationStereo after capture")
        subprocess.run(["fuser", "-k", f"{DEPTH_PORT}/tcp"], capture_output=True)
        time.sleep(4)

    if "propose" in stages:
        cmd = PIXI + ["-m", "gwm_hardware.gwm_arm.propose",
                      "--h5-path", wrist_h5, "--output-dir", proposals,
                      "--k-total", args.k_total]
        if not args.use_plane_normal:
            cmd.append("--horizontal-cut")
        run(cmd, "propose")

    if "score" in stages:
        # Drop any requested view this capture does not carry, rather than
        # failing the run: a camera can be uncalibrated or unplugged, and one
        # good scoring view is still a working rig.
        import h5py

        if external_h5.exists():
            with h5py.File(external_h5) as f:
                have = [c for c in args.cam.split(",") if c.strip() in f]
            missing = [c for c in args.cam.split(",") if c.strip() not in have]
            if missing:
                _log.warning(f"{external_h5.name} has no {missing}; scoring from {have}")
            if not have:
                raise SystemExit(f"{external_h5} carries none of the requested cameras "
                                 f"({args.cam}); groups present: see h5ls")
            args.cam = ",".join(have)
        if not external_h5.exists():
            raise SystemExit(
                f"{external_h5} missing. GWM scores from the third-person view; without "
                "it there is nothing to score. Capture one with\n"
                "  python -m gwm_hardware.gwm_arm.capture live --external-only "
                f"--out-dir {run_dir}"
            )
        run(PIXI + ["-m", "gwm_tiptop.score_client",
                    "--proposals-dir", proposals, "--external-h5", external_h5,
                    "--instruction", args.instruction, "--cam", args.cam,
                    "--tag", tag, "--rat-scale", args.rat_scale,
                    "--object-score", args.object_score,
                    "--server-url", args.server_url,
                    "--dump-dir", run_dir / f"rat_{tag}"], "score")

    if "gate" in stages:
        cmd = PIXI + ["-m", "gwm_tiptop.grasp_gate",
                      "--proposals-dir", proposals, "--h5-path", wrist_h5]
        # Keep the gate's scene decomposition identical to the proposer's.
        if args.use_plane_normal:
            cmd.append("--use-plane-normal")
        cmd.append("--use-robot-arm-filter")
        if args.gate_min_slab_pts is not None:
            cmd += ["--min-slab-pts", args.gate_min_slab_pts]
        if (proposals / f"scores_{tag}.json").exists():
            cmd += ["--apply", tag]
        else:
            _log.warning(f"no scores_{tag}.json: running the gate as a report only "
                         "(it re-picks WITHIN the object the scorer chose, so it has "
                         "nothing to apply to yet)")
        run(cmd, "gate")

    if "viz" in stages:
        cmd = PIXI + ["-m", "gwm_hardware.gwm_arm.viz_debug",
                      "--proposals-dir", proposals, "--h5-path", wrist_h5,
                      "--tag", tag, "--instruction", args.instruction]
        if external_h5.exists():
            cmd += ["--external-h5", external_h5, "--cam", args.cam]
        if not args.rerun:
            cmd.append("--no-rerun")
        if not args.use_plane_normal:
            cmd.append("--horizontal-cut")
        run(cmd, "viz")

    winner = proposals / f"winner_{tag}.json"
    if "execute" in stages:
        if not winner.exists():
            raise SystemExit(f"{winner} missing -- the score stage has not run for this tag")
        # run_real only ever drives PICK plans, which start from an open gripper.
        cmd = PIXI + ["-m", "gwm_hardware.gwm_arm.execute", "--plan", winner, "--open-before"]
        if args.execute:
            cmd.append("--execute")
        if args.go_to_start:
            cmd.append("--go-to-start")
        if args.yes:
            cmd.append("--yes")
        run(cmd, "execute")

    summary = {"run_dir": str(run_dir), "instruction": args.instruction, "tag": tag,
               "stages": stages, "executed": bool(args.execute and "execute" in stages)}
    scores_path = proposals / f"scores_{tag}.json"
    if scores_path.exists():
        sc = json.loads(scores_path.read_text())
        summary["selected_target"] = sc.get("selected_target")
        summary["winner_file"] = sc.get("winner_file")
        summary["object_ranking"] = sc.get("object_ranking")
    (run_dir / f"run_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
