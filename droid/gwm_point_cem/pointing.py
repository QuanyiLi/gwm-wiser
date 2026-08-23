"""pointing: IK, hover-candidate trajectories, and cached scoring for gwm_point_cem.

A candidate is a straight joint-space ride from the home pose to an IK
solution whose fingertip hovers at (x, y, Z_HOVER) with the tool axis
pointing straight down and the yaw frozen at the home yaw; uniform duration
and waypoint count, gripper closed throughout, no gripper event.

Scoring goes through gwm_tiptop.score_client.score_candidates against both
external cameras, fused by the mean. Every (lattice point, instruction) pair
is scored at most once per camera via the JSON-backed ScoreCache.
"""

import json
import sys

import numpy as np

from config import (CAMS, CAPTURE_DIR, GRIPPER, LATTICE_STEP, REPO, SERVER_URL,
                    TRAJ_DURATION, TRAJ_STEPS, URDF, Z_HOVER)

sys.path.insert(0, str(REPO / "droid"))
sys.path.insert(0, str(REPO))

from gwm_tiptop.score_client import score_candidates  # noqa: E402

# Robotiq 2F-85 driver values (franka_renderer.MIMIC_MAP x DRIVER_RANGE_RAD),
# articulation active-joint order after the 7 arm joints:
# [left_outer_knuckle, left_inner_knuckle, right_outer_knuckle,
#  right_inner_knuckle, left_inner_finger, right_inner_finger]
_GRIP_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
_DRIVER_RANGE_RAD = 0.8


def load_views(h5_path=None):
    """external_obs.h5 -> ({cam: (rgb, K, c2w)}, q_init). Same maths as score_client.main."""
    import h5py
    from scipy.spatial.transform import Rotation

    h5_path = h5_path or CAPTURE_DIR / "external_obs.h5"
    views = {}
    with h5py.File(h5_path) as f:
        for cam in CAMS:
            pos = np.asarray(f[f"{cam}/pos_w"])
            w, x, y, z = np.asarray(f[f"{cam}/quat_w_ros"])
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            c2w[:3, 3] = pos
            views[cam] = (np.asarray(f[f"{cam}/rgb"])[..., :3],
                          np.asarray(f[f"{cam}/intrinsic_matrix"]), c2w)
        q_init = np.asarray(f["arm_joint_pos"]).ravel()[:7].astype(np.float64)
    return views, q_init


class PointerKinematics:
    """Fingertip-down IK on the scoring URDF via SAPIEN's Pinocchio model."""

    def __init__(self, q_home):
        import sapien

        self.scene = sapien.Scene()
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(URDF))
        links = [l.name for l in self.robot.get_links()]
        self.i8 = links.index("panda_link8")
        self.i_pads = [links.index("left_inner_finger_pad"),
                       links.index("right_inner_finger_pad")]
        self.pm = self.robot.create_pinocchio_model()
        self.dof = self.robot.dof
        self.mask = np.zeros(self.dof)
        self.mask[:7] = 1
        self.qlim = np.asarray(self.robot.get_qlimits())[:7]

        self.q_home = np.asarray(q_home, dtype=np.float64)
        self._full_home = self.full_q(self.q_home)

        # Tool geometry at the home pose: tip offset along the link8 z axis
        # (pads midpoint), and the home yaw the targets keep.
        self.pm.compute_forward_kinematics(self._full_home)
        p8 = self.pm.get_link_pose(self.i8)
        R8 = _quat_to_mat(p8.q)
        pads = np.mean([self.pm.get_link_pose(i).p for i in self.i_pads], axis=0)
        self.d_tip = float(np.dot(pads - p8.p, R8[:, 2]))
        xw = R8[:, 0]
        self.yaw_home = float(np.arctan2(xw[1], xw[0]))
        self._R_down = _rotz(self.yaw_home) @ np.diag([1.0, -1.0, -1.0])  # Rz(yaw) @ Rx(pi)
        self._q_down = _mat_to_quat(self._R_down)
        self._ik_cache: dict = {}

    def full_q(self, q_arm, grip: float = GRIPPER) -> np.ndarray:
        q = np.zeros(self.dof)
        q[:7] = q_arm
        q[7:13] = _GRIP_SIGNS * (grip * _DRIVER_RANGE_RAD)
        return q

    def ik(self, x: float, y: float):
        """(x, y) -> 7-dof qpos with the fingertip at (x, y, Z_HOVER), or None."""
        import sapien

        key = (round(x, 4), round(y, 4))
        if key in self._ik_cache:
            return self._ik_cache[key]
        target_p = np.array([x, y, Z_HOVER + self.d_tip])
        pose = sapien.Pose(target_p, self._q_down)
        sol, ok, err = self.pm.compute_inverse_kinematics(
            self.i8, pose, initial_qpos=self._full_home, active_qmask=self.mask,
            max_iterations=400)
        result = None
        if ok:
            self.pm.compute_forward_kinematics(sol)
            perr = float(np.linalg.norm(self.pm.get_link_pose(self.i8).p - target_p))
            in_lim = np.all(sol[:7] >= self.qlim[:, 0] - 1e-6) and \
                np.all(sol[:7] <= self.qlim[:, 1] + 1e-6)
            if perr < 0.002 and in_lim:
                result = np.asarray(sol[:7], dtype=np.float64)
        self._ik_cache[key] = result
        return result

    def candidate(self, x: float, y: float):
        """Hover candidate dict for gwm-server, or None when IK fails."""
        q_t = self.ik(x, y)
        if q_t is None:
            return None
        pos = np.linspace(self.q_home, q_t, TRAJ_STEPS)
        t = np.linspace(0.0, TRAJ_DURATION, TRAJ_STEPS)
        return {
            "positions": [list(map(float, p)) for p in pos],
            "t": [round(float(v), 4) for v in t],
            "gripper": [GRIPPER] * TRAJ_STEPS,
            "grasp_close_t": None,
        }


def snap(x: float, y: float):
    return (round(round(x / LATTICE_STEP) * LATTICE_STEP, 4),
            round(round(y / LATTICE_STEP) * LATTICE_STEP, 4))


class ScoreCache:
    """(lattice point, instruction) -> per-cam raw score + prior; JSON-backed."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    @staticmethod
    def key(pt, instruction):
        return f"{pt[0]:.4f},{pt[1]:.4f}|{instruction}"

    def get(self, pt, instruction):
        return self.data.get(self.key(pt, instruction))

    def put(self, pt, instruction, entry):
        self.data[self.key(pt, instruction)] = entry

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


def score_points(views, kin, pts, instruction, cache, server_url=SERVER_URL,
                 rat_scale=3.0, task_image="current", chunk=64, log=print):
    """Score lattice points against one instruction on both cameras, fused by mean.

    Returns {pt: {"score","prior","per_cam"}}; IK-infeasible points map to None.
    Only cache misses touch the server; the cache is saved after every chunk.
    """
    out, todo = {}, []
    for pt in pts:
        hit = cache.get(pt, instruction)
        if hit is not None:
            out[pt] = hit
        elif kin.candidate(*pt) is None:
            out[pt] = None
        else:
            todo.append(pt)
    todo = list(dict.fromkeys(todo))
    sampling = {"rat_scale": rat_scale, "task_image": task_image}
    for lo in range(0, len(todo), chunk):
        batch = todo[lo:lo + chunk]
        cands = [kin.candidate(*pt) for pt in batch]
        per_cam = {}
        for cam in CAMS:
            rgb, K, c2w = views[cam]
            r = score_candidates(server_url, rgb, K, c2w, instruction, cands,
                                 dict(sampling))
            per_cam[cam] = {"scores": r["scores"], "priors": r["stats"]["priors"]}
        for i, pt in enumerate(batch):
            entry = {
                "score": float(np.mean([per_cam[c]["scores"][i] for c in CAMS])),
                "prior": float(np.mean([per_cam[c]["priors"][i] for c in CAMS])),
                "per_cam": {c: {"score": per_cam[c]["scores"][i],
                                "prior": per_cam[c]["priors"][i]} for c in CAMS},
            }
            cache.put(pt, instruction, entry)
            out[pt] = entry
        cache.save()
        log(f"    scored {min(lo + chunk, len(todo))}/{len(todo)} new points")
    return out


def _quat_to_mat(q):
    from scipy.spatial.transform import Rotation
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _mat_to_quat(m):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
