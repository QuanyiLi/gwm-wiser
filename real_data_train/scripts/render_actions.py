"""Offline robot-only rendering: source episodes -> the normalized rendered tree.

For every admitted episode-stream this renders EVERY frame (no temporal
subsampling — frame-step decisions belong to training, decision D-24) with the
shared Franka renderer and writes:

    <data_root>/rendered/<source>/<clip_id>/
        robot_only.mkv                    one video per clip, native res —
                                          FFV1 lossless bit-exact for real
                                          sources (D-27); near-lossless VP9
                                          for molmobot (D-32) — verified per
                                          clip at write time
        meta.json                         alignment + provenance record

meta.json carries everything training needs to pair the robot-only stream
with the source RGB (video path, frame offset, timestamps) plus the full
render provenance (URDF, mount fit, camera parameters), so the training side
never touches source-specific formats.

MolmoBot: the arm mount is recovered per episode from tcp_pose by least
squares and doubles as a kinematics gate (residual <= 2 mm or the stream is
rejected). MolmoAct2-DROID (decision D-28): calibration comes from the KarlP
join (scripts/prepare_droid_calibration.py, run first); every stream must
pass the edge-alignment gate (renderer/edge_gate.py), and keep_ranges idle
filtering is materialized here — each non-idle range long enough to hold a
window becomes its own clip (<ep>__<cam>__seg<k>), idle frames are never
rendered.

Sharding for slurm arrays: --shard-index/--num-shards split the deterministic
episode-stream list; shards are disjoint so array tasks never collide.

Examples:

    python -m real_data_train.scripts.render_actions --source molmobot
    python -m real_data_train.scripts.render_actions --source molmobot \\
        --shard-index $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SCHEMA_VERSION = 2   # v2: per-clip FFV1 video (D-27); v1 was PNG per frame


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
    p.add_argument("--edge-min-score", type=float, default=None,
                   help="DROID edge-gate lift floor "
                        "(default: edge_gate.DEFAULT_MIN_SCORE)")
    p.add_argument("--edge-min-margin", type=float, default=None,
                   help="DROID edge-gate true-minus-perturbed floor "
                        "(default: edge_gate.DEFAULT_MIN_MARGIN)")
    return p.parse_args(argv)


# The shortest renderable keep-range: one full RAT window (2.95 s) plus the
# schedule tolerance, at DROID's 15 fps.
MIN_SEGMENT_FRAMES = 45


def segment_keep_ranges(keep_ranges, n_frames: int,
                        min_len: int = MIN_SEGMENT_FRAMES) -> list:
    """Non-idle [start, end) ranges clipped to the episode, long enough for
    at least one window. keep_ranges=None (episode missing from the KarlP
    filter file) keeps the whole episode — Round-2 decision: lenient."""
    if keep_ranges is None:
        ranges = [(0, n_frames)]
    else:
        ranges = [(max(0, int(s)), min(n_frames, int(e)))
                  for s, e in keep_ranges]
    return [(s, e) for s, e in ranges if e - s >= min_len]


def _clip_done(clip_dir: Path, n_frames: int) -> bool:
    meta_path = clip_dir / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        recorded = meta["n_frames"]
    except (json.JSONDecodeError, KeyError):
        return False
    if recorded != n_frames:
        return False
    return (clip_dir / meta.get("robot_only_video", "robot_only.mkv")).is_file()


def _write_clip(clip_dir: Path, frames: np.ndarray, meta: dict,
                fps: float, codec: str = "ffv1") -> None:
    """Write robot_only.mkv (verified) then meta.json (completion mark).

    codec="ffv1": bit-exact lossless, the real-source contract (D-27).
    codec="vp9": near-lossless VP9 for the high-volume sim tree (D-32),
    verified against the calibrated tolerance gates. Every clip write is its
    own verification gate either way.
    """
    from real_data_train import lossless_video as lv

    clip_dir.mkdir(parents=True, exist_ok=True)
    video_path = clip_dir / "robot_only.mkv"
    if codec == "ffv1":
        lv.write_lossless_video(frames, video_path, fps=fps)
        lv.verify_lossless(video_path, frames)
        codec_tag = "ffv1/bgr0"
    elif codec == "vp9":
        lv.write_near_lossless_video(frames, video_path, fps=fps)
        lv.verify_near_lossless(video_path, frames)
        codec_tag = lv.NEAR_LOSSLESS_CODEC
    else:
        raise ValueError(f"unknown clip codec: {codec}")
    meta = {**meta, "robot_only_video": "robot_only.mkv",
            "robot_only_codec": codec_tag}
    tmp_meta = clip_dir / "meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=1))
    tmp_meta.rename(clip_dir / "meta.json")  # meta.json last = completion mark


def render_molmobot(args, provenance: dict) -> dict:
    import sapien

    from real_data_train.renderer.assets import build_welded_urdf
    from real_data_train.renderer.franka_renderer import (
        FrankaRobotRenderer,
        fit_arm_mount,
    )
    from real_data_train.sources import molmobot

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
        _write_clip(clip_dir, frames, meta, fps=1.0 / ep.dt_s, codec="vp9")
        stats["rendered"] += 1
        logging.info("rendered %s (%d frames, mount rms %.2f mm)",
                     ep.clip_id, ep.n_frames, rms * 1000)
    return stats


def render_molmoact2_droid(args, provenance: dict) -> dict:
    from real_data_train.lossless_video import read_video_frames
    from real_data_train.renderer import edge_gate
    from real_data_train.renderer.assets import build_welded_urdf
    from real_data_train.renderer.franka_renderer import FrankaRobotRenderer
    from real_data_train.sources import molmoact2_droid

    min_score = (edge_gate.DEFAULT_MIN_SCORE if args.edge_min_score is None
                 else args.edge_min_score)
    min_margin = (edge_gate.DEFAULT_MIN_MARGIN if args.edge_min_margin is None
                  else args.edge_min_margin)

    cameras = tuple(args.cameras) if args.cameras else molmoact2_droid.CAMERAS
    calibrations = molmoact2_droid.load_calibrations(args.data_root)
    episodes, excluded = molmoact2_droid.discover_episodes(
        args.data_root / "molmoact2_droid", cameras=cameras,
        calibrations=calibrations,
    )
    streams = sorted((e for e in episodes if e.calibrated),
                     key=lambda e: e.clip_id)
    shard = streams[args.shard_index::args.num_shards]
    if args.limit is not None:
        shard = shard[:args.limit]
    logging.info("molmoact2_droid: %d streams on disk, %d calibrated, "
                 "%d in shard", len(episodes), len(streams), len(shard))
    if not calibrations and episodes:
        logging.warning("no calibration.json — run "
                        "scripts/prepare_droid_calibration.py first")

    urdf = build_welded_urdf("panda", args.data_root / "assets")
    renderer = FrankaRobotRenderer(urdf, arm="panda")
    w, h = molmoact2_droid.VIDEO_WH
    out_root = args.data_root / "rendered" / "molmoact2_droid"

    stats = {"rendered_clips": 0, "skipped_done": 0, "rejected_edge_gate": 0,
             "no_segments": 0, "streams_done": 0, "discovered": len(episodes),
             "calibrated": len(streams), "excluded_discovery": len(excluded)}
    for ep in shard:
        calib = ep.calibration
        segments = segment_keep_ranges(calib["keep_ranges"], ep.length)
        if not segments:
            stats["no_segments"] += 1
            continue
        seg_dirs = [out_root / f"{ep.clip_id}__seg{k}"
                    for k in range(len(segments))]
        if not args.overwrite and all(
                _clip_done(d, e - s)
                for d, (s, e) in zip(seg_dirs, segments)):
            stats["skipped_done"] += 1
            continue

        st = molmoact2_droid.load_states(ep)
        intr = calib["intrinsics"]
        c2w = calib["cam2world_cv"]

        def render_probe(cam2world, probes, st=st, intr=intr):
            return renderer.render(
                st["arm_qpos"][probes], st["gripper_pos"][probes],
                intr, cam2world, w, h,
            )

        probes = edge_gate.probe_indices(
            [i for s, e in segments for i in range(s, e)])
        observed = read_video_frames(
            ep.video_path, [ep.video_frame_start + p for p in probes])
        scores = edge_gate.gate_scores(
            lambda cw: render_probe(cw, probes), observed, c2w)
        if not edge_gate.passes(scores, min_score, min_margin):
            logging.warning("%s: edge gate rejected (score %.3f, "
                            "perturbed %.3f)", ep.clip_id,
                            scores["score"], scores["score_perturbed"])
            stats["rejected_edge_gate"] += 1
            continue

        for k, ((s, e), clip_dir) in enumerate(zip(segments, seg_dirs)):
            if not args.overwrite and _clip_done(clip_dir, e - s):
                continue
            frames = renderer.render(
                st["arm_qpos"][s:e], st["gripper_pos"][s:e], intr, c2w, w, h,
            )
            meta = {
                "schema_version": SCHEMA_VERSION,
                "source": "molmoact2_droid",
                "clip_id": f"{ep.clip_id}__seg{k}",
                "episode_uid": ep.episode_uid,
                "camera": ep.camera,
                "n_frames": e - s,
                "width": w,
                "height": h,
                "timestamps": [i / molmoact2_droid.FPS for i in range(s, e)],
                "rgb_video": str(ep.video_path.relative_to(args.data_root)),
                "rgb_frame_start": ep.video_frame_start + s,
                "keep_range": [s, e],
                "intrinsics": np.asarray(intr).tolist(),
                "extrinsic_cam2base_6d": calib["extrinsic_cam2base_6d"],
                "camera_serial": calib["serial"],
                "calibration_quality": calib["quality_metric"],
                "extrinsic_source": calib["extrinsic_source"],
                "droid_episode_id": calib["episode_id"],
                "edge_gate": {**scores, "probe_frames": probes},
                "task_type": "droid",
                "task_description": st["language_instruction"],
                "renderer_provenance": {"arm": "panda", "urdf": str(urdf),
                                        **provenance},
            }
            _write_clip(clip_dir, frames, meta, fps=molmoact2_droid.FPS)
            stats["rendered_clips"] += 1
        stats["streams_done"] += 1
        logging.info("rendered %s: %d segments, edge score %.3f (vs %.3f)",
                     ep.clip_id, len(segments), scores["score"],
                     scores["score_perturbed"])
    return stats


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from real_data_train.renderer.assets import ensure_source_repos, provenance

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
        stats = render_molmoact2_droid(args, prov)
        logging.info("molmoact2_droid: %s", stats)


if __name__ == "__main__":
    main()
