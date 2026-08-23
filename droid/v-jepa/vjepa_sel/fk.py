"""Forward kinematics of the Franka Panda arm to `panda_link8` (the flange).

Modified (Craig) DH parameters from Franka's documentation; the frame returned
is panda_link8 in the robot base frame, which is the frame DROID reports in
`robot_state/cartesian_position` (polymetis EE link `panda_link8`) and hence
the frame V-JEPA 2-AC's state channel was trained on. The Robotiq gripper
hangs below it; its geometry does not enter the state.

Validated against the Isaac-sim articulation's `panda_link8` world pose in
sim/validate_fk.py (the sim robot base sits at the world origin).
"""

import numpy as np
from scipy.spatial.transform import Rotation

# (a, d, alpha) per joint, modified DH; theta = q_i
_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_FLANGE = (0.0, 0.107, 0.0)  # link7 -> link8 (flange), theta = 0


def _tf(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_link8(q):
    """q [7] -> 4x4 base_T_link8."""
    T = np.eye(4)
    for (a, d, alpha), th in zip(_DH, q):
        T = T @ _tf(a, d, alpha, th)
    return T @ _tf(*_FLANGE, 0.0)


def fk_link8_batch(qs):
    """qs [N, 7] -> positions [N, 3], rotation matrices [N, 3, 3]."""
    qs = np.asarray(qs, dtype=np.float64)
    pos = np.zeros((len(qs), 3))
    rot = np.zeros((len(qs), 3, 3))
    for i, q in enumerate(qs):
        T = fk_link8(q)
        pos[i] = T[:3, 3]
        rot[i] = T[:3, :3]
    return pos, rot


def rot_to_euler_xyz(R):
    """Rotation matrix -> scipy extrinsic 'xyz' Euler (DROID's cartesian_position convention)."""
    return Rotation.from_matrix(R).as_euler("xyz", degrees=False)


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()
