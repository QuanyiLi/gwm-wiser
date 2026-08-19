"""One decomposition of a capture, shared by everything that reads it.

The proposer, the grasp gate and the debug viewer each start from the same
wrist h5 and each independently redo the same work: load it, project depth to
a world cloud, RANSAC the table, DBSCAN the clusters. Measured on the zhiwei
rig that is ~1.1 s a time, three times a turn, for a byte-identical answer.

Keying on the file's path, size and mtime means a re-capture invalidates it
automatically while a re-read never pays twice, and the cache is per-process,
so nothing persists across runs to go stale on disk.

The decomposition FLAGS are part of the key on purpose. The gate judging plans
against a scene decomposed differently from the one the proposer planned in is
not a hypothetical -- it happened on 2026-08-19, the gate lost the target
object, and every candidate for it came back "no raw points". Two callers
asking for different flags get different entries and no illusion of agreement.
"""

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_CACHE: dict = {}
_MAX_ENTRIES = 4        # a session revisits the current capture, not a history


def scene(h5_path, use_plane_normal: bool = False, robot_spheres=None,
          above_table: float = 0.015, depth_clip=(0.05, 4.0)) -> dict:
    """Decompose a wrist capture once.

    Returns obs, the structured world cloud, the fitted table and the clusters.
    `robot_spheres` participates in the key by its byte content, since two
    different arm configurations decompose the scene differently.
    """
    from tiptop.perception.utils import depth_to_xyz

    from gwm_tiptop.perception_geometric import cluster_objects, find_table_plane
    from gwm_tiptop.propose_from_h5 import load_h5_observation

    path = Path(h5_path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size, bool(use_plane_normal),
           float(above_table), tuple(depth_clip),
           None if robot_spheres is None else hash(np.asarray(robot_spheres).tobytes()))
    hit = _CACHE.get(key)
    if hit is not None:
        _log.debug(f"scene cache hit for {path.name}")
        return hit

    obs = load_h5_observation(path)
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= depth_clip[0]) | (depth > depth_clip[1])] = np.nan
    valid_frac = float(np.isfinite(depth).mean())
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0

    table_box, surface_z = find_table_plane(xyz_map, rgb_map)
    meshes, pcds = cluster_objects(xyz_map, rgb_map, table_box, surface_z + above_table,
                                   use_plane_normal=use_plane_normal,
                                   robot_spheres=robot_spheres)
    out = {"obs": obs, "depth": depth, "valid_frac": valid_frac,
           "xyz_map": xyz_map, "rgb_map": rgb_map,
           "table_box": table_box, "surface_z": surface_z,
           "meshes": meshes, "pcds": pcds}

    if len(_CACHE) >= _MAX_ENTRIES:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def clear() -> None:
    _CACHE.clear()
