"""Offline robot-only rendering: source episodes -> the normalized rendered tree.

For every admitted episode-stream this renders EVERY frame (no temporal
subsampling — frame-step decisions belong to training, decision D-24) with the
shared Franka renderer and writes:

    <data_root>/rendered/<source>/<clip_id>/
        robot_only/00000.png ...          lossless, native mp4 resolution
        meta.json                         alignment + provenance record

meta.json carries everything training needs to pair the robot-only stream
with the source RGB (video path, frame offset, timestamps) plus the full
render provenance (URDF, mount fit, camera parameters), so the training side
never touches source-specific formats.

MolmoBot: the arm mount is recovered per episode from tcp_pose by least
squares and doubles as a kinematics gate (residual <= 2 mm or the stream is
rejected). MolmoAct2-DROID: streams stay uncalibrated until the DROID
camera-recovery gate lands; they are enumerated and skipped with a count.

Sharding for slurm arrays: --shard-index/--num-shards split the deterministic
episode-stream list; shards are disjoint so array tasks never collide.

Examples:

    python -m real_world_gwm.scripts.render_actions --source molmobot
    python -m real_world_gwm.scripts.render_actions --source molmobot \\
        --shard-index $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SCHEMA_VERSION = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--source", choices=["molmobot", "molmoact2_droid", "all"],
                   default="all")
    p.add_argument("--configs", nargs="+", default=None,
                   help="molmobot config filter")
    p.add_argument("--split", default=None, help="molmobot split filter")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N streams (smoke use)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mount-max-residual-m", type=float, default=0.002)
    return p.parse_args(argv)


def _clip_done(clip_dir: Path, n_frames: int) -> bool:
    meta = clip_dir / "meta.json"
    if not meta.is_file():
        return False
    try:
        recorded = json.loads(meta.read_text())["n_frames"]
    except (json.JSONDecodeError, KeyError):
        return False
    return recorded == n_frames and (
        len(list((clip_dir / "robot_only").glob("*.png"))) == n_frames
    )


def _write_clip(clip_dir: Path, frames: np.ndarray, meta: dict) -> None:
    from PIL import Image

    tmp_meta = clip_dir / "meta.json.tmp"
    frame_dir = clip_dir / "robot_only"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"{i:05d}.png")
    tmp_meta.write_text(json.dumps(meta, indent=1))
    tmp_meta.rename(clip_dir / "meta.json")  # meta.json last = completion mark


def render_molmobot(args, provenance: dict) -> dict:
    import sapien

    from real_world_gwm.renderer.assets import build_welded_urdf
    from real_world_gwm.renderer.franka_renderer import (
        FrankaRobotRenderer,
        fit_arm_mount,
    )
    from real_world_gwm.sources import molmobot

    urdf = build_welded_urdf("fr3", args.data_root / "assets")
    renderer = FrankaRobotRenderer(urdf, arm="fr3")

    cameras = tuple(args.cameras) if args.cameras else molmobot.EXTERIOR_CAMERAS
    episodes, excluded = molmobot.discover_episodes(
        args.data_root, configs=args.configs, split=args.split, cameras=cameras
    )
    episodes.sort(key=lambda e: e.clip_id)
    shard = episodes[args.shard_index::args.num_shards]
    if args.limit is not None:
        shard = shard[:args.limit]

    stats = {"rendered": 0, "skipped_done": 0, "rejected_mount": 0,
             "discovered": len(episodes), "excluded_discovery": len(excluded)}
    out_root = args.data_root / "rendered" / "molmobot"
    w, h = molmobot.MP4_WH
    for ep in shard:
        clip_dir = out_root / ep.clip_id
        if not args.overwrite and _clip_done(clip_dir, ep.n_frames):
            stats["skipped_done"] += 1
            continue
        st = molmobot.load_states(ep)
        try:
            mount, tcp_dz, rms = fit_arm_mount(
                renderer, st["arm_qpos"], st["gripper_qpos"], st["tcp_pose"],
                max_residual_m=args.mount_max_residual_m,
            )
        except RuntimeError as err:
            logging.warning("%s: %s", ep.clip_id, err)
            stats["rejected_mount"] += 1
            continue
        arm_base = np.stack([
            np.concatenate([
                (sapien.Pose(b[:3], b[3:]) * sapien.Pose(mount)).p,
                (sapien.Pose(b[:3], b[3:]) * sapien.Pose(mount)).q,
            ])
            for b in st["base_pose"]
        ])
        frames = renderer.render(
            st["arm_qpos"], st["gripper_qpos"], st["intrinsics_mp4"],
            st["cam2world_gl"], w, h, base_pose=arm_base,
        )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "source": "molmobot",
            "clip_id": ep.clip_id,
            "episode_uid": ep.episode_uid,
            "camera": ep.camera,
            "n_frames": ep.n_frames,
            "width": w,
            "height": h,
            "timestamps": ep.timestamps(),
            "rgb_video": str(ep.mp4_path.relative_to(args.data_root)),
            "rgb_frame_start": 0,
            "fov_deg": st["fov_deg"],
            "intrinsics": np.asarray(st["intrinsics_mp4"]).tolist(),
            "mount": {"translation": np.asarray(mount).tolist(),
                      "tcp_dz": tcp_dz, "fit_rms_m": rms},
            "task_type": st["task_type"],
            "task_description": st["task_description"],
            "renderer_provenance": {"arm": "fr3", "urdf": str(urdf),
                                    **provenance},
        }
        _write_clip(clip_dir, frames, meta)
        stats["rendered"] += 1
        logging.info("rendered %s (%d frames, mount rms %.2f mm)",
                     ep.clip_id, ep.n_frames, rms * 1000)
    return stats


def render_molmoact2_droid(args) -> dict:
    from real_world_gwm.sources import molmoact2_droid

    cameras = tuple(args.cameras) if args.cameras else molmoact2_droid.CAMERAS
    episodes, excluded = molmoact2_droid.discover_episodes(
        args.data_root, cameras=cameras
    )
    calibrated = [e for e in episodes if e.calibrated]
    logging.warning(
        "molmoact2_droid: %d episode-streams on disk, %d calibrated — "
        "rendering requires the DROID camera-recovery gate (plan of record); "
        "uncalibrated streams are skipped",
        len(episodes), len(calibrated),
    )
    if calibrated:
        raise NotImplementedError(
            "calibrated DROID rendering lands with the camera-recovery gate"
        )
    return {"discovered": len(episodes), "calibrated": 0, "rendered": 0,
            "excluded_discovery": len(excluded)}


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from real_world_gwm.renderer.assets import ensure_source_repos, provenance

    ensure_source_repos(args.data_root / "assets")
    prov = {"assets": provenance(args.data_root / "assets")}
    try:
        import sapien

        prov["sapien"] = sapien.__version__
    except ImportError:
        pass

    if args.source in ("molmobot", "all"):
        stats = render_molmobot(args, prov)
        logging.info("molmobot: %s", stats)
    if args.source in ("molmoact2_droid", "all"):
        stats = render_molmoact2_droid(args)
        logging.info("molmoact2_droid: %s", stats)


if __name__ == "__main__":
    main()
