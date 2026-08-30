"""Sim-free checks of the kinematics and the task math (python tests/test_kin.py)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwm_rl import franka_kin as K  # noqa: E402
from gwm_rl import geometry as G  # noqa: E402
from gwm_rl.task import TaskParams, grasp_alignment, grasp_depth, pick_reward, rim_geometry  # noqa: E402


def test_home_pose_points_down():
    p, R = K.fk_tcp(K.HOME_Q[None])
    assert torch.allclose(R[0], K.R_DOWN, atol=1e-6)
    assert abs(K.yaw_of(R)[0].item()) < 1e-6
    assert 0.3 < p[0, 0] < 0.4 and abs(p[0, 1]) < 1e-6 and 0.3 < p[0, 2] < 0.4


def test_ik_reaches_a_top_down_target():
    q = K.HOME_Q[None].clone()
    target = torch.tensor([[0.45, 0.05, 0.25]])
    yaw = torch.tensor([0.3])
    for _ in range(60):
        q = K.ik_step(q, target, K.target_rotation(yaw))
    p, R = K.fk_tcp(q)
    assert (p - target).norm() < 1e-4
    assert K.rotvec_from_matrix(K.target_rotation(yaw) @ R.transpose(-1, -2)).norm() < 1e-4


def test_yaw_interpolates_modulo_pi():
    assert abs(K.wrap_half_turn(torch.tensor(math.pi - 0.1)).item() - (-0.1)) < 1e-6
    assert abs(K.wrap_half_turn(torch.tensor(-math.pi + 0.1)).item() - 0.1) < 1e-6
    p0 = torch.zeros(1, 3)
    _, yaw = K.interpolate_pose(p0, torch.tensor([math.pi / 2 - 0.01]), p0, torch.tensor([-math.pi / 2]), 4)
    assert yaw[0, -1].item() > math.pi / 2 - 0.02  # the short way: +0.01, not a half turn


def test_plan_macro_ends_at_target_and_holds():
    q0 = K.HOME_Q.expand(3, 7).clone()
    target = torch.tensor([[0.40, 0.05, 0.12]]).expand(3, 3)
    yaw = torch.full((3,), 0.5)
    traj = K.plan_macro(q0, target, yaw, n_move=30, n_hold=14)
    assert traj.shape == (3, 44, 7)
    p, _ = K.fk_tcp(traj[:, -1])
    assert (p - target).norm(dim=-1).max() < 1e-3
    assert torch.allclose(traj[:, 29], traj[:, -1])
    assert (traj[:, 1:] - traj[:, :-1]).abs().max() <= 0.12 + 1e-6


def test_rim_geometry_upright_bowl():
    bowl_pos = torch.tensor([list(G.BOWL_POS)])
    bowl_quat = torch.tensor([list(G.BOWL_QUAT)])
    tcp = torch.tensor([[G.BOWL_POS[0] - 0.3, G.BOWL_POS[1], 0.3]])
    rim, radial, axis = rim_geometry(tcp, bowl_pos, bowl_quat)
    assert torch.allclose(axis[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)  # rim on top
    assert abs(rim[0, 2].item() - (G.BOWL_POS[2] + G.BOWL_HALF_HEIGHT)) < 1e-4
    assert abs(rim[0, 0].item() - (G.BOWL_POS[0] - G.BOWL_RIM_RADIUS)) < 1e-4
    assert torch.allclose(radial[0], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-4)
    R = K.target_rotation(torch.tensor([math.pi / 2]))  # pads (flange y) along world x
    assert grasp_alignment(R, radial, axis)[0].item() > 0.999
    R = K.target_rotation(torch.tensor([0.0]))  # pads across the radial
    assert abs(grasp_alignment(R, radial, axis)[0].item() - 0.5) < 1e-4
    below = rim.clone(); below[0, 2] -= 0.012
    assert abs(grasp_depth(below, rim, axis)[0].item() - 0.012) < 1e-6
    assert abs(grasp_depth(tcp, rim, axis)[0].item() + (0.3 - (G.BOWL_POS[2] + G.BOWL_HALF_HEIGHT))) < 1e-4


def test_reward_is_staged():
    p = TaskParams()
    t, f = torch.tensor([True]), torch.tensor([False])
    z, o = torch.zeros(1), torch.ones(1)

    def r(**kw):
        base = dict(grasp_dist=z, grasp_align=o, grasp_depth_ok=o, is_grasped=f, height_reached=f, bowl_height=z,
                    height_margin=z, dwell_fraction=z, success=f, obstacle_force=z, params=p)
        base.update(kw)
        return pick_reward(**base)[0].item()

    far = r(grasp_dist=torch.tensor([0.4]))
    near = r()
    pinch = r(is_grasped=t, grasp_depth_ok=z)  # a pinch on the rim's top edge
    grasped = r(is_grasped=t)
    lifting = r(is_grasped=t, bowl_height=torch.tensor([0.08]))
    held = r(is_grasped=t, height_reached=t, dwell_fraction=torch.tensor([0.5]))
    success = r(is_grasped=t, height_reached=t, dwell_fraction=o, success=t, height_margin=torch.tensor([0.05]))
    assert far < near < pinch < grasped < lifting < held < success
    assert abs(grasped - r(is_grasped=t, bowl_height=z)) < 1e-9  # no pay for zero lift
    assert 1.0 <= success * 10 <= 13.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
