"""MolmoBot-Data source reader (extracted scene packages).

Reads the layout produced by scripts/setup_data.py (mirroring the authors'
bulk_download.py):

    <data_root>/molmobot/<Config>/part<P>/<split>/<house_id>/
        trajectories_batch_<B>.h5
        episode_<8d>_<camera>_batch_<B>.mp4     (624x352, ~15.15 fps)

h5 facts (checked on FrankaPickOmniCamConfig val):
    traj_<i>/obs/agent/qpos          (T, 2000) uint8 null-padded JSON:
                                     {"arm": [7], "base": [], "gripper": [2]}
    traj_<i>/obs/sensor_param/<cam>/{intrinsic_cv (T,3,3), cam2world_gl (T,4,4),
                                     extrinsic_cv (T,3,4)}
    traj_<i>/obs_scene               scalar JSON with policy_dt_ms, task info

Intrinsics: the h5 ``intrinsic_cv`` (480x480, c=(240,240)) belongs to a
different internal render and must NOT be used for the released mp4s. The
authoritative per-camera parameters live in ``obs_scene.frozen_config``
(camera_config.cameras[*].fov, vertical, degrees; img_resolution 624x352),
so K_mp4 = [[f, 0, 312], [0, f, 176], [0, 0, 1]] with
f = 176 / tan(fov / 2). Verified by URDF re-projection overlay:
this K + cam2world_gl aligns the rendered robot with the released mp4, while
the intrinsic_cv-derived candidates land off-scale.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Admitted exterior streams: the GoPro analogue
# (137-140 deg FOV) is held out, the wrist camera is never used.
EXTERIOR_CAMERAS = (
    "droid_shoulder_light_randomization",
    "randomized_zed2_analogue_1",
    "randomized_zed2_analogue_2",
)

MP4_WH = (624, 352)


def decode_json_bytes(datum) -> object:
    return json.loads(bytes(datum).decode("utf-8").rstrip("\x00"))


def fov_intrinsics(fov_deg: float, width: int = 624, height: int = 352) -> np.ndarray:
    """Pinhole K from a vertical FOV (MuJoCo convention), centered."""
    import math

    f = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


class _StubConfig:
    """Placeholder for unpickled config classes (authors' safe pattern)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__ = state["__dict__"] if (
            isinstance(state, dict) and "__dict__" in state) else (
            state if isinstance(state, dict) else {})


class _ConfigUnpickler(__import__("pickle").Unpickler):
    def find_class(self, module, name):
        if module.startswith(("numpy", "pathlib")):
            import importlib

            return getattr(importlib.import_module(module), name)
        return _StubConfig


def camera_fovs(obs_scene: dict) -> dict:
    """{camera_name: vertical fov (deg)} from the episode's frozen config."""
    import base64
    import io

    cfg = _ConfigUnpickler(
        io.BytesIO(base64.b64decode(obs_scene["frozen_config"]))
    ).load()
    cameras = cfg.camera_config.cameras
    return {c.__dict__["name"]: float(c.__dict__["fov"]) for c in cameras}


@dataclass
class MolmoBotEpisode:
    config: str
    house: str
    h5_path: Path
    traj_key: str          # e.g. "traj_0"
    traj_idx: int
    batch_suffix: str      # e.g. "batch_1_of_1"
    camera: str
    mp4_path: Path
    n_frames: int
    dt_s: float

    @property
    def episode_uid(self) -> str:
        """Camera-independent id: both streams of one episode share it."""
        return f"molmobot/{self.config}/{self.house}/{self.batch_suffix}_ep{self.traj_idx}"

    @property
    def clip_id(self) -> str:
        return (f"{self.config}__{self.house}__"
                f"{self.batch_suffix}_ep{self.traj_idx}__{self.camera}")

    def timestamps(self) -> list:
        return [i * self.dt_s for i in range(self.n_frames)]


def _iter_h5_files(root: Path, configs=None, split=None):
    base = Path(root) / "molmobot"
    if not base.is_dir():
        return
    for config_dir in sorted(base.iterdir()):
        if not config_dir.is_dir():
            continue
        if configs and config_dir.name not in configs:
            continue
        for h5_path in sorted(config_dir.glob("part*/*/*/trajectories_*.h5")):
            if split and h5_path.parent.parent.name != split:
                continue
            yield config_dir.name, h5_path


def discover_episodes(
    root, configs=None, split=None, cameras=EXTERIOR_CAMERAS
) -> tuple:
    """(episodes, excluded) across extracted scene packages."""
    import h5py

    episodes, excluded = [], []
    for config, h5_path in _iter_h5_files(root, configs, split):
        house = h5_path.parent.name
        m = re.match(r"trajectories_(batch_.+)\.h5$", h5_path.name)
        batch_suffix = m.group(1)
        with h5py.File(h5_path, "r") as f:
            if "valid_traj_mask" in f:
                valid = list(np.asarray(f["valid_traj_mask"][()], dtype=bool))
            else:
                idxs = {int(k.split("_")[-1]) for k in f if k.startswith("traj_")}
                valid = [i in idxs for i in range(max(idxs) + 1)]
            for tid, ok in enumerate(valid):
                key = f"traj_{tid}"
                if not ok or key not in f:
                    if not ok:
                        excluded.append({"h5": str(h5_path), "traj": key,
                                         "reason": "invalid_traj_mask"})
                    continue
                traj = f[key]
                scene = decode_json_bytes(traj["obs_scene"][()])
                dt_s = float(scene["policy_dt_ms"]) / 1000.0
                n_frames = traj["obs/agent/qpos"].shape[0]
                for cam in cameras:
                    if cam not in traj["obs/sensor_param"]:
                        excluded.append({"h5": str(h5_path), "traj": key,
                                         "camera": cam,
                                         "reason": "no_sensor_param"})
                        continue
                    mp4 = h5_path.parent / (
                        f"episode_{tid:08d}_{cam}_{batch_suffix}.mp4"
                    )
                    if not mp4.is_file():
                        excluded.append({"h5": str(h5_path), "traj": key,
                                         "camera": cam, "reason": "no_mp4"})
                        continue
                    episodes.append(MolmoBotEpisode(
                        config=config, house=house, h5_path=h5_path,
                        traj_key=key, traj_idx=tid, batch_suffix=batch_suffix,
                        camera=cam, mp4_path=mp4, n_frames=n_frames, dt_s=dt_s,
                    ))
    return episodes, excluded


def load_states(episode: MolmoBotEpisode) -> dict:
    """Per-frame robot state and camera parameters for one episode-stream."""
    import h5py

    with h5py.File(episode.h5_path, "r") as f:
        traj = f[episode.traj_key]
        qpos_raw = traj["obs/agent/qpos"][()]
        arm = np.stack([
            np.asarray(decode_json_bytes(row)["arm"], dtype=np.float64)
            for row in qpos_raw
        ])
        gripper = np.stack([
            np.asarray(decode_json_bytes(row)["gripper"], dtype=np.float64)
            for row in qpos_raw
        ])
        cam = traj["obs/sensor_param"][episode.camera]
        cam2world_gl = np.asarray(cam["cam2world_gl"][()])
        base_pose = np.asarray(traj["obs/extra/robot_base_pose"][()])
        tcp_pose = np.asarray(traj["obs/extra/tcp_pose"][()])
        scene = decode_json_bytes(traj["obs_scene"][()])
    fov = camera_fovs(scene)[episode.camera]
    return {
        "arm_qpos": arm,                       # (T, 7)
        "gripper_qpos": gripper,               # (T, 2) driver joint values
        "fov_deg": fov,                        # vertical, frozen_config
        "intrinsics_mp4": fov_intrinsics(fov), # (3, 3), 624x352 pixels
        "cam2world_gl": cam2world_gl,          # (T, 4, 4) OpenGL convention
        "base_pose": base_pose,                # (T, 7) xyz + wxyz quat, world
        "tcp_pose": tcp_pose,                  # (T, 7) base-local TCP pose
        "task_description": scene.get("task_description"),
        "task_type": scene.get("task_type"),
    }
