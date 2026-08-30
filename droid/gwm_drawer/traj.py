"""traj: IK candidate trajectories for gwm_drawer.

A drawer candidate rides joint space through a fixed waypoint chain:

    home -> stage (16 cm back, 6 cm up from the knob, tool already aligned)
         -> pre   (10 cm back along the tool axis)
         -> grasp (knob between the finger pads)
         -> [gripper-close pause, 20/15 s]
         -> pull  (straight out along the cabinet's slide axis)
         -> hold  (drawer open, gripper closed)

The chain is built in the cabinet's local frame (approach along its front
normal, pitched down) and mapped through the cabinet yaw. One grasp
orientation family (pad-straddle axis) and approach pitch is shared by the
three drawer candidates: the (family, pitch) combination is picked by the
lowest total chain cost over joint-limit margin, chain smoothness, tip-path
quality and clearance to the cabinet boxes.

An object candidate (distractor) uses the same chain shape with a top-down
tool: stage/pre above the object, grasp at the object, close, then a straight
lift instead of the pull. Every timeline totals 8.85 s = the rat_scale-3.0
RAT window; three of the five rendered RAT frames sit at or after the
pull/lift. Every waypoint is solved with SAPIEN's Pinocchio IK on the
scoring URDF; object positions come from the capture's objects.json.

Run with the repo venv:

    /root/code/gwm/gwm-wiser/.venv/bin/python traj.py
"""

import json

import numpy as np

from config import (CAPTURE_DIR, DRAWERS, GRASP_LIFT, OBJECTS, RESULTS,
                    TABLE_TOP_Z, URDF)

# timeline (s): reach 2.4 (stage 1.6) / insert 1.0 / close 20/15 / pull 2.0 /
# hold to 8.85
T_STAGE, T_REACH, T_INSERT, T_PULL = 1.6, 2.4, 1.0, 2.0
T_CLOSE = 20 / 15.0
CLOSE_SUBSTEPS = 7
T_TOTAL = 8.85
N_STAGE, N_PRE, N_INSERT, N_PULL = 9, 5, 6, 8

D_TIP = 0.150                      # link8 -> pads-mid along the tool axis (open)
TIP_EXT = 0.019                    # pads-mid -> fingertip
FRONT_CLEAR = 0.004                # fingertip stays this far off the front panel
STAGE_BACK, STAGE_UP = 0.16, 0.06
PRE_BACK = 0.10

# Orientation families (pad-straddle axis, a cabinet-local vector = link8 y)
# and approach pitches (deg about local y; positive tilts the tool axis down),
# searched jointly for one shared (family, pitch).
FAMILIES = [("pads_y+", (0, 1, 0)), ("pads_y-", (0, -1, 0)),
            ("pads_z+", (0, 0, 1)), ("pads_z-", (0, 0, -1))]
PITCHES = [12.0, 25.0]

ARM_R, HAND_R, FINGER_R = 0.058, 0.036, 0.022


def _rotm(family_y, pitch_deg):
    from scipy.spatial.transform import Rotation
    y8 = np.asarray(family_y, dtype=float)
    z8 = np.array([1.0, 0.0, 0.0])
    x8 = np.cross(y8, z8)
    R = np.column_stack([x8, y8, z8])
    return Rotation.from_euler("y", pitch_deg, degrees=True).as_matrix() @ R


class PandaKin:
    """FK/IK on the scoring URDF via SAPIEN's Pinocchio model."""

    def __init__(self, q_home):
        import sapien

        self.scene = sapien.Scene()
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(URDF))
        self.links = [l.name for l in self.robot.get_links()]
        self.i8 = self.links.index("panda_link8")
        self.i_pads = [self.links.index("left_inner_finger_pad"),
                       self.links.index("right_inner_finger_pad")]
        self.i_arm = [self.links.index(f"panda_link{k}") for k in range(8)]
        self.pm = self.robot.create_pinocchio_model()
        self.dof = self.robot.dof
        self.mask = np.zeros(self.dof)
        self.mask[:7] = 1
        self.qlim = np.asarray(self.robot.get_qlimits())[:7]
        self.q_home = np.asarray(q_home, dtype=np.float64)

    def full_q(self, q_arm, grip=0.0):
        q = np.zeros(self.dof)
        q[:7] = q_arm
        q[7:13] = np.array([1, 1, 1, 1, -1, -1]) * (grip * 0.8)
        return q

    def ik(self, target_p, R, seeds, iters=3000):
        import sapien
        from scipy.spatial.transform import Rotation

        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        pose = sapien.Pose(np.asarray(target_p), np.array([qw, qx, qy, qz]))
        for seed in seeds:
            sol, ok, _ = self.pm.compute_inverse_kinematics(
                self.i8, pose, initial_qpos=self.full_q(seed),
                active_qmask=self.mask, max_iterations=iters)
            if not ok:
                continue
            self.pm.compute_forward_kinematics(sol)
            perr = np.linalg.norm(np.asarray(self.pm.get_link_pose(self.i8).p) - target_p)
            Rs = Rotation.from_quat(np.roll(self.pm.get_link_pose(self.i8).q, -1)).as_matrix()
            aerr = np.degrees(np.arccos(np.clip((np.trace(Rs.T @ R) - 1) / 2, -1, 1)))
            q = np.asarray(sol[:7], dtype=np.float64)
            if perr < 0.0025 and aerr < 1.5 and self.limit_margin(q) > 0.05:
                return q
        return None

    def limit_margin(self, q):
        return float(min((q - self.qlim[:, 0]).min(), (self.qlim[:, 1] - q).min()))

    def tip(self, q_arm):
        self.pm.compute_forward_kinematics(self.full_q(q_arm))
        return np.mean([self.pm.get_link_pose(i).p for i in self.i_pads], axis=0)

    def sample_points(self, q_arm):
        """(point, radius, cls) samples along the arm, hand and fingers."""
        self.pm.compute_forward_kinematics(self.full_q(q_arm))
        P = [np.asarray(self.pm.get_link_pose(i).p) for i in range(len(self.links))]
        pts = []
        arm = [P[i] for i in self.i_arm]
        for a, b in zip(arm[:-1], arm[1:]):
            for s in np.linspace(0, 1, 4):
                pts.append((a + s * (b - a), ARM_R, "arm"))
        i_base = self.links.index("robotiq_arg2f_base_link")
        for a, b in [(P[self.i8], P[i_base])]:
            for s in np.linspace(0, 1, 3):
                pts.append((a + s * (b - a), HAND_R, "hand"))
        for i in self.i_pads:
            pts.append((np.asarray(self.pm.get_link_pose(i).p), FINGER_R, "finger"))
        pm_mid = np.mean([self.pm.get_link_pose(i).p for i in self.i_pads], axis=0)
        from scipy.spatial.transform import Rotation
        R8 = Rotation.from_quat(np.roll(self.pm.get_link_pose(self.i8).q, -1)).as_matrix()
        pts.append((pm_mid + TIP_EXT * R8[:, 2], FINGER_R, "finger"))
        return pts


def cab_aabbs():
    """World AABB of every cabinet's (possibly yawed) footprint."""
    boxes = []
    for cab in DRAWERS.values():
        R, center = cab.frame()
        hx, hy = cab.depth / 2, cab.width / 2
        corners = np.array([(sx * hx, sy * hy, 0) for sx in (-1, 1) for sy in (-1, 1)])
        world = (R @ corners.T).T + center
        lo = world.min(axis=0) + [0, 0, cab.z0]
        hi = world.max(axis=0) + [0, 0, cab.z0 + cab.height]
        boxes.append((cab.name, lo, hi))
    return boxes


def clearance_violations(kin, q, exempt_centers, exempt_r=0.13):
    """Sampled robot points vs cabinet AABBs and the table plane; hand/finger
    points within exempt_r of any exempt center (the grasp/pull region of the
    target) are excused from the cabinet check."""
    bad = []
    ex = [np.asarray(c) for c in exempt_centers]
    for p, r, cls in kin.sample_points(q):
        if cls in ("hand", "finger") and any(
                np.linalg.norm(p - c) < exempt_r for c in ex):
            continue
        if p[2] - r < 0.052 and p[0] > 0.08:
            bad.append((cls, "table", tuple(np.round(p, 3))))
        for name, lo, hi in cab_aabbs():
            d = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
            if np.linalg.norm(d) < r:
                bad.append((cls, name, tuple(np.round(p, 3))))
    return bad


def solve_chain(kin, key, family, pitch, rng):
    cab = DRAWERS[key]
    R, center = cab.frame()
    Rb = _rotm(family, pitch)
    z8l = Rb[:, 2]
    kxl, kyl, kzl = cab.knob_local()
    fxl = cab.front_x_local
    # pads-mid at grasp: fingertip FRONT_CLEAR off the front panel plane
    pm_xl = fxl - FRONT_CLEAR - TIP_EXT * z8l[0]
    grasp_l = np.array([pm_xl, kyl, kzl + (pm_xl - kxl) * z8l[2] / max(z8l[0], 1e-6)])
    pre_l = grasp_l - PRE_BACK * z8l
    stage_l = grasp_l - STAGE_BACK * z8l + np.array([0, 0, STAGE_UP])
    pulls_l = [grasp_l + np.array([-cab.pull * s, 0, 0])
               for s in np.linspace(1 / 4, 1, 4)]

    def w(p_l):
        return R @ np.asarray(p_l) + center

    R0 = R @ Rb
    z8 = R0[:, 2]
    wps = [w(stage_l), w(pre_l), w(grasp_l)] + [w(p) for p in pulls_l]

    seeds0 = [kin.q_home]
    for _ in range(14):
        seeds0.append(np.clip(kin.q_home + rng.normal(0, 0.7, 7),
                              kin.qlim[:, 0] + 0.1, kin.qlim[:, 1] - 0.1))
    chain, seeds = [], seeds0
    for wp in wps:
        q = kin.ik(wp - D_TIP * z8, R0, seeds)
        if q is None:
            return None
        chain.append(q)
        seeds = [q]

    margins = [kin.limit_margin(q) for q in chain]
    steps = [float(np.abs(b - a).max()) for a, b in zip(chain[:-1], chain[1:])]
    if max(steps) > 1.0:
        return None
    exempt = wps[2:]
    qs_screen = list(chain)
    for a, b in zip([kin.q_home] + chain[:-1], chain):
        qs_screen.append((np.asarray(a) + np.asarray(b)) / 2)
    for q in qs_screen:
        if clearance_violations(kin, q, exempt):
            return None

    # Tip-path quality of the joint-space ride home -> stage -> pre: penalize
    # arcs that balloon upward or wander far beyond the direct line.
    tips = []
    for a, b in [(kin.q_home, chain[0]), (chain[0], chain[1])]:
        for s in np.linspace(0, 1, 10):
            tips.append(kin.tip(np.asarray(a) + s * (np.asarray(b) - np.asarray(a))))
    tips = np.asarray(tips)
    apex = float(tips[:, 2].max())
    path_len = float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum())
    wander = path_len / max(float(np.linalg.norm(tips[-1] - tips[0])), 0.05)
    cost = (2.0 * max(steps) + 0.5 * sum(steps)
            + 0.5 * float(np.abs(chain[0] - kin.q_home).max())
            - min(min(margins), 0.3)
            + 8.0 * max(0.0, apex - 0.55) + 0.8 * max(0.0, wander - 1.15))
    return {"chain": chain, "cost": cost, "min_margin": min(margins),
            "max_step": max(steps), "pitch": pitch}


def solve_grasp(kin, key, obj_pos, rng):
    """Top-down grasp-and-lift chain on a table object (distractor candidate):
    home -> stage (16 cm above) -> pre (10 cm above) -> grasp -> close -> lift."""
    from scipy.spatial.transform import Rotation

    spec = OBJECTS[key]
    # tool z down; finger pads open along the world direction `straddle`
    R0 = (Rotation.from_euler("z", spec["straddle"] + 90.0, degrees=True).as_matrix()
          @ np.diag([1.0, -1.0, -1.0]))
    z8 = R0[:, 2]
    grasp = np.array([obj_pos[0] - spec["radial"], obj_pos[1],
                      TABLE_TOP_Z + spec["z_above"]])
    up = np.array([0.0, 0.0, 1.0])
    wps = [grasp + STAGE_BACK * up, grasp + PRE_BACK * up, grasp] + \
          [grasp + GRASP_LIFT * s * up for s in np.linspace(1 / 4, 1, 4)]

    seeds0 = [kin.q_home]
    for _ in range(14):
        seeds0.append(np.clip(kin.q_home + rng.normal(0, 0.7, 7),
                              kin.qlim[:, 0] + 0.1, kin.qlim[:, 1] - 0.1))
    chain, seeds = [], seeds0
    for wp in wps:
        q = kin.ik(wp - D_TIP * z8, R0, seeds)
        if q is None:
            return None
        chain.append(q)
        seeds = [q]
    margins = [kin.limit_margin(q) for q in chain]
    steps = [float(np.abs(b - a).max()) for a, b in zip(chain[:-1], chain[1:])]
    if max(steps) > 1.0:
        return None
    qs_screen = list(chain)
    for a, b in zip([kin.q_home] + chain[:-1], chain):
        qs_screen.append((np.asarray(a) + np.asarray(b)) / 2)
    for q in qs_screen:
        if clearance_violations(kin, q, wps[2:]):
            return None
    return {"chain": chain, "min_margin": min(margins), "max_step": max(steps),
            "grasp": grasp.tolist()}


def solve_all(kin):
    """Drawers: one (family, pitch) shared by all three, lowest total chain
    cost. Objects: one top-down grasp chain each."""
    combos = []
    for fam_name, fam_y in FAMILIES:
        for pitch in PITCHES:
            sols = {}
            for key in DRAWERS:
                sol = solve_chain(kin, key, fam_y, pitch, np.random.default_rng(7))
                if sol is None:
                    break
                sol["family"] = fam_name
                sols[key] = sol
            if len(sols) == len(DRAWERS):
                combos.append((sum(s["cost"] for s in sols.values()), sols))
    if not combos:
        raise SystemExit("no family+pitch solves all drawer candidates")
    sols = min(combos, key=lambda c: c[0])[1]

    objects = json.loads((CAPTURE_DIR / "objects.json").read_text())
    for key, spec in OBJECTS.items():
        pos = objects[spec["prim"]]["pos_w"]
        sol = solve_grasp(kin, key, pos, np.random.default_rng(7))
        if sol is None:
            raise SystemExit(f"no grasp chain for {key}")
        sols[key] = sol
    return sols


def timeline(kin, chain):
    """Waypoint chain -> scoring-candidate dict (positions, t, gripper)."""
    t_hold = T_TOTAL - T_REACH - T_INSERT - T_CLOSE - T_PULL
    q_stage, q_pre, q_grasp = chain[0], chain[1], chain[2]
    pulls = chain[3:]
    P, T, G = [], [], []

    def seg(q_from, q_to, t0, t1, n, g):
        for s in np.linspace(0, 1, n + 1)[1:]:
            P.append(q_from + s * (q_to - q_from))
            T.append(t0 + s * (t1 - t0))
            G.append(g)

    P.append(kin.q_home)
    T.append(0.0)
    G.append(0.0)
    t = 0.0
    seg(kin.q_home, q_stage, t, t + T_STAGE, N_STAGE, 0.0)
    t += T_STAGE
    seg(q_stage, q_pre, t, t + (T_REACH - T_STAGE), N_PRE, 0.0)
    t += T_REACH - T_STAGE
    seg(q_pre, q_grasp, t, t + T_INSERT, N_INSERT, 0.0)
    t += T_INSERT
    close_t = t
    for k in range(CLOSE_SUBSTEPS):
        P.append(q_grasp)
        T.append(t + T_CLOSE * (k + 1) / CLOSE_SUBSTEPS)
        G.append((k + 1) / CLOSE_SUBSTEPS)
    t += T_CLOSE
    n_per = max(1, N_PULL // len(pulls))
    q_prev = q_grasp
    dt = T_PULL / len(pulls)
    for q_next in pulls:
        seg(q_prev, q_next, t, t + dt, n_per, 1.0)
        t += dt
        q_prev = q_next
    for tt in (t + t_hold / 2, t + t_hold):
        P.append(q_prev)
        T.append(tt)
        G.append(1.0)
    return {
        "positions": [list(map(float, q)) for q in P],
        "t": [round(float(x), 4) for x in T],
        "gripper": [round(float(g), 4) for g in G],
        "grasp_close_t": round(float(close_t), 4),
    }


def main() -> None:
    import h5py

    with h5py.File(CAPTURE_DIR / "external_obs.h5") as f:
        q_home = np.asarray(f["arm_joint_pos"]).ravel()[:7].astype(np.float64)
    kin = PandaKin(q_home)
    RESULTS.mkdir(exist_ok=True)

    solutions = solve_all(kin)
    out = {}
    for key, best in solutions.items():
        cand = timeline(kin, best["chain"])
        if key in DRAWERS:
            cab = DRAWERS[key]
            meta = {"kind": "drawer", "cabinet": cab.name, "yaw": cab.yaw,
                    "knob": [round(v, 4) for v in cab.knob_center()],
                    "pull": cab.pull, "family": best["family"],
                    "pitch": best["pitch"]}
        else:
            meta = {"kind": "grasp", "object": OBJECTS[key]["prim"],
                    "grasp": [round(v, 4) for v in best["grasp"]],
                    "lift": GRASP_LIFT}
        meta.update({"min_limit_margin": round(best["min_margin"], 3),
                     "max_wp_step": round(best["max_step"], 3)})
        out[key] = {"candidate": cand, "meta": meta}
        print(f"{key}: {meta['kind']} margin={best['min_margin']:.2f} "
              f"max_step={best['max_step']:.2f} T={cand['t'][-1]}s")

    (RESULTS / "candidates.json").write_text(json.dumps(out))
    print(f"wrote {RESULTS / 'candidates.json'}")


if __name__ == "__main__":
    main()
