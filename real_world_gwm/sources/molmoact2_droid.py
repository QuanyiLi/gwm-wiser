"""MolmoAct2-DROID source reader (LeRobot v3.0 repo files on disk).

Reads the partial snapshot produced by scripts/setup_data.py:

    <data_root>/molmoact2_droid/
        meta/info.json, meta/episodes/chunk-000/file-*.parquet
        data/chunk-000/file-*.parquet          (multi-episode, 100 MB target)
        videos/observation.images.<cam>/chunk-000/file-*.mp4   (concatenated)

Discovery intersects the episodes metadata with what is actually on disk, so
any subset download yields exactly the episodes that are complete locally.

Camera calibration status (verified 2026-08-06): the per-frame
``camera_extrinsics.*`` columns exist but are zero-filled across the entire
release (meta/stats.json: min = max = mean = 0), and no intrinsics are
published. Episode-streams therefore carry ``calibrated=False`` until the
DROID camera-recovery gate (plan of record) supplies verified per-episode
parameters; the render pipeline refuses uncalibrated streams.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

CAMERAS = ("exterior_1_left", "exterior_2_left")   # wrist is never used
FPS = 15.0


@dataclass
class DroidEpisode:
    root: Path
    episode_index: int
    camera: str
    data_path: Path            # parquet holding this episode's rows
    video_path: Path           # concatenated mp4 holding this episode
    video_from_ts: float       # episode start within the video file (s)
    length: int                # frames
    calibrated: bool = False
    calibration: dict = field(default=None)

    @property
    def episode_uid(self) -> str:
        return f"molmoact2_droid/ep{self.episode_index:06d}"

    @property
    def clip_id(self) -> str:
        return f"ep{self.episode_index:06d}__{self.camera}"

    @property
    def video_frame_start(self) -> int:
        return int(round(self.video_from_ts * FPS))

    def timestamps(self) -> list:
        return [i / FPS for i in range(self.length)]


def _episode_meta_tables(root: Path):
    import pyarrow.parquet as pq

    for p in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        yield pq.read_table(p, columns=[
            "episode_index", "length", "data/chunk_index", "data/file_index",
            *[f"videos/observation.images.{c}/{k}"
              for c in CAMERAS for k in
              ("chunk_index", "file_index", "from_timestamp")],
        ]).to_pandas()


def discover_episodes(root, cameras=CAMERAS, calibrations: dict = None) -> tuple:
    """(episodes, excluded) for everything complete on disk.

    calibrations: optional {episode_uid/camera: {"intrinsics": 3x3,
    "extrinsics_pose": [x,y,z,r,p,y]}} mapping from the (future) DROID
    camera-recovery gate; streams without an entry stay uncalibrated.
    """
    root = Path(root) / "molmoact2_droid" if not (
        Path(root).name == "molmoact2_droid") else Path(root)
    episodes, excluded = [], []
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a molmoact2_droid tree: {root}")

    for df in _episode_meta_tables(root):
        for r in df.to_dict("records"):
            ep = int(r["episode_index"])
            data_path = (root / "data" /
                         f"chunk-{int(r['data/chunk_index']):03d}" /
                         f"file-{int(r['data/file_index']):03d}.parquet")
            if not data_path.is_file():
                continue  # outside the downloaded subset: silently absent
            for cam in cameras:
                vprefix = f"videos/observation.images.{cam}"
                video_path = (root / "videos" / f"observation.images.{cam}" /
                              f"chunk-{int(r[f'{vprefix}/chunk_index']):03d}" /
                              f"file-{int(r[f'{vprefix}/file_index']):03d}.mp4")
                if not video_path.is_file():
                    excluded.append({"episode_index": ep, "camera": cam,
                                     "reason": "video_file_absent"})
                    continue
                key = f"molmoact2_droid/ep{ep:06d}/{cam}"
                calib = (calibrations or {}).get(key)
                episodes.append(DroidEpisode(
                    root=root, episode_index=ep, camera=cam,
                    data_path=data_path, video_path=video_path,
                    video_from_ts=float(r[f"{vprefix}/from_timestamp"]),
                    length=int(r["length"]),
                    calibrated=calib is not None,
                    calibration=calib,
                ))
    return episodes, excluded


@lru_cache(maxsize=4)
def _data_table(data_path: str):
    import pyarrow.parquet as pq

    return pq.read_table(data_path, columns=[
        "episode_index", "frame_index", "timestamp",
        "observation.state", "language_instruction",
    ]).to_pandas()


def load_states(episode: DroidEpisode) -> dict:
    """Per-frame robot state for one episode (camera-independent)."""
    df = _data_table(str(episode.data_path))
    g = df[df["episode_index"] == episode.episode_index].sort_values("frame_index")
    state = np.stack(g["observation.state"].values)   # (T, 8): 7 joints + gripper
    return {
        "arm_qpos": state[:, :7].astype(np.float64),
        "gripper_pos": state[:, 7].astype(np.float64),  # continuous [0, 1]
        "timestamps": g["timestamp"].to_numpy(dtype=np.float64),
        "language_instruction": (
            g["language_instruction"].iloc[0] if len(g) else None
        ),
    }
