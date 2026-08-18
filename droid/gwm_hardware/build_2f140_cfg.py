"""Generate the cuRobo robot config (.yml) for Panda + Robotiq 2F-140.

Companion to `build_2f140.py`, which emits the URDF. This produces the cuRobo
side: collision spheres, self-collision table, locked/mimic joints and cspace.

The arm half is inherited verbatim from cuTAMP's `panda_robotiq_2f_85.yml`
(same Panda, same tuned sphere set, same buffers), so only the gripper half is
new. Gripper spheres are **fitted to the 2F-140 collision meshes by cuRobo's
own `Mesh.get_bounding_spheres`** rather than hand-placed: hand-placing 2F-85
numbers onto 140 mm fingers is exactly the failure this whole exercise exists
to avoid, and a mesh fit is reproducible and auditable.

One thing does NOT carry over automatically. The DROID wrist-camera collision
volume is not a link in the 2F-85 model -- it is baked into
`robotiq_85_base_link`'s sphere list as two rows of 0.025 m spheres. Those are
re-emitted here against `robotiq_140_base_link`, but their placement was tuned
for the 2F-85 + DROID mount, so on this rig they are a **placeholder that must
be re-measured against the actual wrist bracket** before trusting the planner
near the table. Guarded by `--camera-spheres/--no-camera-spheres`.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.build_2f140_cfg
"""

import argparse
from pathlib import Path

import yaml

ASSETS = Path(__file__).resolve().parent / "assets"
CUTAMP_ASSETS = Path(__file__).resolve().parents[1] / "tiptop/cutamp/cutamp/robots/assets"
SRC_YML = CUTAMP_ASSETS / "panda_robotiq_2f_85.yml"
URDF_NAME = "panda_robotiq_2f_140.urdf"

ARM_LINKS = [f"panda_link{i}" for i in range(8)]

# Gripper links that get collision geometry, and how many spheres each gets.
# Budgets scale with the part's volume; the fingers get the most because they
# are what actually approaches the object and the table.
# Ceilings, not targets: _cover_with_spheres stops at the first pitch that
# meets MAX_THICKNESS_INFLATION and only spends more spheres if it has to.
# Raised on 2026-08-18 -- the old values were far below what the old fitter
# actually emitted (the 8-sphere pad budget produced 55), so they constrained
# nothing, while a tight fit on the plate-like links genuinely needs the room.
GRIPPER_SPHERE_BUDGET = {
    "robotiq_140_base_link": 200,
    "left_outer_knuckle": 200,
    "right_outer_knuckle": 200,
    "left_outer_finger": 200,
    "right_outer_finger": 200,
    "left_inner_knuckle": 200,
    "right_inner_knuckle": 200,
    "left_inner_finger": 200,
    "right_inner_finger": 200,
    "left_inner_finger_pad": 200,
    "right_inner_finger_pad": 200,
}

# Wrist-camera mount.
#
# Derived from the hand-eye calibration (2026-08-18) rather than guessed. The
# 2F-85 config carries two rows of spheres on ONE side of the gripper base;
# copying that verbatim was a hazard, because the bracket is the one asymmetric
# thing on the wrist and if it is modelled on the wrong side cuRobo guards empty
# air while the real camera meets the table unprotected. It was mirrored to both
# sides as an interim measure; the calibration now says which side it is.
#
#   camera optical centre, in robotiq_140_base_link:
#       [-65.4, +36.8, +65.9] mm      (i.e. the +y side)
#   optical axis 1.0 deg off the gripper approach axis
#
# What gets enclosed: a D435 body is roughly 90 x 25 x 25 mm, long axis along
# its stereo baseline (camera +x). That box is swept from the optical centre,
# and a second run of spheres bridges back to the gripper base to cover the
# bracket, whose geometry is not modelled anywhere.
CAMERA_BODY_LEN_M = 0.090
CAMERA_BODY_R_M = 0.022          # half-thickness plus margin for the housing
CAMERA_BRACKET_R_M = 0.020


def camera_mount_spheres(urdf_path: Path):
    """Spheres enclosing the wrist camera and its bracket, in the gripper base
    frame, placed from the measured hand-eye extrinsic."""
    import numpy as np
    from yourdfpy import URDF

    from tiptop.config import load_calibration, tiptop_cfg

    cfg = tiptop_cfg()
    ee_from_cam = load_calibration(str(cfg.cameras.hand.serial))

    robot = URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)
    robot.update_cfg({**{f"panda_joint{i+1}": 0.0 for i in range(7)},
                      DRIVER_JOINT: 0.0})
    base_from_cam = robot.get_transform("grasp_frame", "robotiq_140_base_link") @ ee_from_cam
    centre = base_from_cam[:3, 3]
    body_axis = base_from_cam[:3, 0]        # camera +x = stereo baseline

    rows = []
    for f in np.linspace(-0.5, 0.5, 5):     # along the camera body
        c = centre + body_axis * (f * CAMERA_BODY_LEN_M)
        rows.append({"center": [round(float(v), 6) for v in c],
                     "radius": CAMERA_BODY_R_M})
    for f in np.linspace(0.25, 1.0, 3):     # bracket, camera back to the base
        c = centre * (1.0 - f)
        rows.append({"center": [round(float(v), 6) for v in c],
                     "radius": CAMERA_BRACKET_R_M})
    return rows


DRIVER_JOINT = "finger_joint"

# A fitted sphere set is only safe if it actually encloses the part. Every
# collision-mesh vertex must sit inside some sphere (inflated by this slack) or
# the planner will happily drive that bit of gripper through an obstacle.
MIN_COVERAGE = 0.98
COVERAGE_SLACK_M = 0.0005

# How much fatter than a part's own thinnest dimension a sphere may be.
#
# This is the one real knob, and it trades safety against planning speed.
# Capping pitch at the exact thickness (1.0) gives a faithful 287-sphere
# gripper that cuRobo plans with 30x slower than the stock 36-sphere 2F-85 --
# and TiPToP refines 16 candidate trajectories per scene, so 30x turns a
# 2-minute proposal pass into an hour. Larger values coarsen thin parts.
# Whatever is chosen, coverage stays 100% by construction; only the
# over-approximation grows. Set from --thickness-multiple and recorded in the
# emitted config's header.
THICKNESS_MULTIPLE = 1.0

# The other half of the check that was missing until 2026-08-18.
#
# MIN_COVERAGE only asks "do the spheres cover the mesh". Fat spheres satisfy
# that trivially, so it never noticed that thin parts were being modelled
# 1.73-2.73x thicker than they are -- the outer finger, 27.1 mm of real
# bracket, came out as a 72.8 mm slab. 1.73 is sqrt(3), the circumradius of a
# cube: the old radius assumed every voxel was FULL, so a plate one voxel thick
# got a sphere sized for a solid cube of that pitch. 2.73 is 1 + sqrt(3), the
# same thing when grid alignment puts two layers across the thickness.
#
# That inflation propagates: build_gripper_spheres derives cuTAMP's grasp
# filter from these spheres, and cuTAMP keeps that envelope clear of the target
# object, so an over-thick gripper pushes every grasp outward. Measured
# consequence -- the executed grasp landed 8-12 mm above EVERY M2T2 proposal on
# all three objects tried, which is survivable on a tomato and fatal on a 77 mm
# open cup, where it puts the pads above the rim.
# 1.80 is a REGRESSION GUARD, not a target -- it sits just above what the
# fitter reaches at a sphere count cuRobo can plan with. The measured trade,
# total gripper spheres against worst-link inflation:
#
#     lateral pitch = thickness      190 spheres    1.42 - 1.78x   <- chosen
#                   = thickness/1.5  806            1.52 - 1.94x
#                   = thickness/2   1305            1.36 - 1.71x
#                   = thickness/3   3241            1.32 - 1.47x
#
# Inflation is not even monotonic in the pitch (grid alignment decides whether
# a layer boundary lands on the surface), and the floor is ~1.3x at 17x the
# spheres. The old cube-circumradius fitter sat at 287 spheres / 1.73 - 2.73x,
# so the chosen point is better on both axes at once; buying the last 0.3x
# would cost more planning time than the tighter model is worth.
MAX_THICKNESS_INFLATION = 1.80
SURFACE_SAMPLES = 40000


def _surface_points(mesh):
    """Dense, deterministic sample of the mesh surface, plus its vertices.

    Vertices alone are a weak test: a coarse mesh has few of them and they sit
    at extremes, so spheres can pass a vertex check while leaving whole faces
    outside.
    """
    import numpy as np

    pts, _ = __import__("trimesh").sample.sample_surface_even(mesh, SURFACE_SAMPLES,
                                                              seed=0)
    return np.vstack([np.asarray(mesh.vertices), np.asarray(pts)])


def _coverage(vertices, rows) -> float:
    """Fraction of collision-mesh sample points enclosed by the fitted spheres."""
    import numpy as np

    centers = np.array([r["center"] for r in rows])
    radii = np.array([r["radius"] for r in rows]) + COVERAGE_SLACK_M
    d = np.linalg.norm(vertices[:, None, :] - centers[None, :, :], axis=-1)
    return float((d <= radii[None, :]).any(axis=1).mean())


def _link_collision_mesh(robot, link_name):
    """Union of a link's collision geometry, in the link frame."""
    import numpy as np
    import trimesh

    parts = []
    for col in robot.link_map[link_name].collisions:
        geom = col.geometry
        if geom.mesh is not None:
            mesh = trimesh.load(robot._filename_handler(geom.mesh.filename),
                                force="mesh")
            if geom.mesh.scale is not None:
                mesh.apply_scale(geom.mesh.scale)
        elif geom.box is not None:
            mesh = trimesh.creation.box(extents=geom.box.size)
        else:
            continue
        if col.origin is not None:
            mesh.apply_transform(col.origin)
        parts.append(mesh)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _cover_with_spheres(mesh, budget: int):
    """Cover a mesh with <= budget spheres, covering **by construction**.

    Voxelise the solid, then put one sphere at each occupied voxel centre with
    the voxel cube's circumradius (pitch * sqrt(3) / 2). The union of those
    spheres provably contains the voxelised solid, so the planner can never see
    through the part -- the failure mode that matters for a gripper.

    cuRobo's own `get_bounding_spheres` is not used: its
    VOXEL_VOLUME_SAMPLE_SURFACE fit spends 25 % of the budget on 2 mm surface
    spheres and packs the rest strictly *inside* the volume, which measured
    2 % vertex coverage on the 2F-140 base link. That is fine for the "grow an
    attached object" use it was written for, and wrong for a robot link.

    Pitch is bisected so the voxel count lands just under the budget.
    """
    import numpy as np
    from trimesh.voxel.creation import voxelize

    def n_at(pitch):
        try:
            vg = voxelize(mesh, pitch, "subdivide").fill("base")
            return np.asarray(vg.points)
        except Exception:
            return np.zeros((0, 3))

    # Cap the pitch at the part's THINNEST dimension. Without this, a budget
    # derived from the long axis inflates thin parts absurdly -- the 7.5 mm
    # finger pad came out as 26 mm-radius spheres, a gripper so fat in the
    # planner's eyes that it collides with everything it tries to grasp.
    # Over-approximating a finger refuses valid grasps just as surely as
    # under-approximating one drives it through the table.
    surf = _surface_points(mesh)
    ext = np.asarray(mesh.extents, dtype=float)
    ax = int(np.argmin(ext))                 # thin axis
    la, lb = [k for k in range(3) if k != ax]
    thin = float(ext[ax])

    def fit(pitch):
        """Anisotropic: a lateral grid, and only as many layers through the
        thin axis as the local material actually needs.

        An isotropic voxel grid pays n^3 to refine when only the lateral n^2
        buys tightness, which is why the cube-circumradius version could not
        get under ~1.4x without ~2000 spheres. Here each lateral cell looks at
        the material actually in it: one sphere centred on that material, sized
        to reach it, and extra layers only where the part is genuinely thick.
        """
        lo = surf.min(0)
        idx = np.floor((surf[:, [la, lb]] - lo[[la, lb]]) / pitch).astype(int)
        rows = []
        for key in np.unique(idx, axis=0):
            m = (idx == key).all(1)
            q = surf[m]
            span = float(q[:, ax].max() - q[:, ax].min())
            layers = max(1, int(np.ceil(span / pitch)))
            edges = np.linspace(q[:, ax].min(), q[:, ax].max(), layers + 1)
            for k in range(layers):
                sel = (q[:, ax] >= edges[k] - 1e-9) & (q[:, ax] <= edges[k + 1] + 1e-9)
                if not sel.any():
                    continue
                blk = q[sel]
                c = np.empty(3)
                c[la] = lo[la] + (key[0] + 0.5) * pitch
                c[lb] = lo[lb] + (key[1] + 0.5) * pitch
                c[ax] = 0.5 * (blk[:, ax].min() + blk[:, ax].max())
                r = float(np.linalg.norm(blk - c, axis=1).max())
                # reach half a cell diagonally so neighbouring spheres still
                # overlap across a cell face, otherwise the seams leak
                rows.append((c, max(r, pitch * (2 ** 0.5) / 2)))
        pts = np.array([c for c, _ in rows])
        rad = np.array([r for _, r in rows])
        # EVERY axis, not just the thin one. Checking only the thin axis is the
        # same one-sided mistake as checking only coverage: the first version of
        # this fitter passed the thin-axis bound while modelling the gripper base
        # as four 106 mm balls, which bulged sideways far enough to put the
        # capture pose in collision with the table.
        env = (pts + rad[:, None]).max(0) - (pts - rad[:, None]).min(0)
        return pts, rad, float((env / ext).max())

    best = None
    for div in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        pts, rad, infl = fit(thin / div)
        if len(pts) > budget:
            break
        best = (pts, rad, infl)
        if infl <= MAX_THICKNESS_INFLATION:
            break
    # (infl below is the worst axis, see fit())
    if best is None:
        raise RuntimeError(
            f"even the coarsest fit exceeds the {budget}-sphere budget; "
            f"raise it in GRIPPER_SPHERE_BUDGET")
    pts, rad, _ = best
    return [{"center": [round(float(c), 6) for c in pt],
             "radius": round(float(r), 6)} for pt, r in zip(pts, rad)]


def fit_gripper_spheres(urdf_path: Path) -> dict:
    """Fit collision spheres to each 2F-140 link's own collision mesh."""
    import numpy as np
    from yourdfpy import URDF

    robot = URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
    out, total = {}, 0
    for link_name, budget in GRIPPER_SPHERE_BUDGET.items():
        mesh = _link_collision_mesh(robot, link_name)
        if mesh is None:
            continue
        rows = _cover_with_spheres(mesh, budget)
        pts = np.array([r["center"] for r in rows])
        rad = np.array([r["radius"] for r in rows])
        samples = _surface_points(mesh)
        cover = _coverage(samples, rows)
        # Two-sided, and both sides matter. Coverage alone is satisfied by any
        # sufficiently fat sphere, which is exactly how 2.7x inflation went
        # unnoticed; inflation alone is satisfied by no spheres at all.
        env3 = (pts + rad[:, None]).max(0) - (pts - rad[:, None]).min(0)
        ratios = env3 / np.asarray(mesh.extents, dtype=float)
        ax = int(np.argmax(ratios))
        env = float(env3[ax])
        inflation = float(ratios[ax])
        out[link_name] = rows
        total += len(rows)
        print(f"  {link_name:26s} {len(rows):3d} spheres  r={rad.min():.4f}-{rad.max():.4f} m  "
              f"coverage {cover * 100:5.1f} %  "
              f"worst axis {'xyz'[ax]} {mesh.extents[ax]*1000:5.1f} -> {env*1000:5.1f} mm  "
              f"({inflation:4.2f}x)")
        if cover < MIN_COVERAGE:
            raise RuntimeError(
                f"{link_name}: fitted spheres cover only {cover * 100:.1f} % of "
                f"the collision mesh surface (need >= {MIN_COVERAGE * 100:.0f} %). "
                f"Raise its budget in GRIPPER_SPHERE_BUDGET.")
        if inflation > MAX_THICKNESS_INFLATION:
            raise RuntimeError(
                f"{link_name}: modelled {inflation:.2f}x fatter than the real part on "
                f"axis {'xyz'[ax]} ({mesh.extents[ax]*1000:.1f} mm -> {env*1000:.1f} mm), over the "
                f"{MAX_THICKNESS_INFLATION:.2f}x bound. An over-thick gripper makes "
                f"cuTAMP push grasps off the object. Raise its budget in "
                f"GRIPPER_SPHERE_BUDGET so a finer pitch fits.")
    print(f"  {'total':26s} {total:3d} spheres")
    return out


def build_cfg(urdf_path: Path, out_path: Path, camera_spheres: bool) -> Path:
    src = yaml.safe_load(SRC_YML.read_text())
    kin = src["robot_cfg"]["kinematics"]

    gripper_links = list(GRIPPER_SPHERE_BUDGET)
    spheres = {k: v for k, v in kin["collision_spheres"].items() if k in ARM_LINKS}
    fitted = fit_gripper_spheres(urdf_path)
    if camera_spheres:
        cam_rows = camera_mount_spheres(urdf_path)
        fitted["robotiq_140_base_link"] = fitted["robotiq_140_base_link"] + cam_rows
        c = cam_rows[0]["center"]
        print(f"  + {len(cam_rows)} wrist-camera spheres on robotiq_140_base_link, "
              f"placed from the hand-eye calibration (+y side)")
    spheres.update(fitted)

    # Self-collision: never check gripper parts against each other or against
    # the wrist links they are rigidly bolted to; the linkage is always in
    # contact with itself by construction.
    ignore = {
        "panda_link0": ["panda_link1", "panda_link2"],
        "panda_link1": ["panda_link2", "panda_link3", "panda_link4"],
        "panda_link2": ["panda_link3", "panda_link4"],
        "panda_link3": ["panda_link4", "panda_link6"],
        "panda_link4": ["panda_link5", "panda_link6", "panda_link7", "panda_link8"],
        "panda_link5": ["panda_link6", "panda_link7", "robotiq_140_base_link"],
        "panda_link6": ["panda_link7", "robotiq_140_base_link"],
        "panda_link7": ["robotiq_140_base_link", *gripper_links],
        "tool0": gripper_links,
    }
    for i, a in enumerate(gripper_links):
        ignore[a] = [b for b in gripper_links[i + 1:]]
    ignore["attached_object"] = gripper_links

    buffer = {l: src["robot_cfg"]["kinematics"]["self_collision_buffer"].get(l, 0.02)
              for l in ARM_LINKS}
    buffer["robotiq_140_base_link"] = 0.01
    for l in gripper_links:
        buffer.setdefault(l, 0.0)
    buffer["attached_object"] = 0.0

    kin.update({
        # Absolute, not URDF_NAME. cuTAMP's load_panda_robotiq_rerun() resolves
        # this against its OWN assets dir (cutamp/robots/assets/), where our
        # 2F-140 does not live, so a relative name makes tiptop-run die on
        # get_robot_rerun(). Both cuRobo's join_path and pathlib drop the prefix
        # when the suffix is absolute, so this is correct for every consumer.
        "urdf_path": str(urdf_path.resolve()),
        "ee_link": "grasp_frame",
        "collision_link_names": [*ARM_LINKS, *gripper_links, "attached_object"],
        "collision_spheres": spheres,
        "self_collision_ignore": ignore,
        "self_collision_buffer": buffer,
        "mesh_link_names": [*ARM_LINKS, *gripper_links],
        # Only the driver is a DOF; every other gripper joint mimics it in the
        # URDF, so cuRobo derives them.
        "lock_joints": {DRIVER_JOINT: 0.0},
        "extra_links": {"attached_object": {
            "parent_link_name": "grasp_frame", "link_name": "attached_object",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_type": "FIXED", "joint_name": "attach_joint"}},
        "cspace": {
            "joint_names": [f"panda_joint{i}" for i in range(1, 8)] + [DRIVER_JOINT],
            # 2F-140 driver range is [0, 0.7] (2F-85's was [0, 0.8]); the
            # retract value is scaled by the same ratio as the 2F-85's 0.6.
            "retract_config": [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.525],
            "null_space_weight": [1] * 8,
            "cspace_distance_weight": [1] * 8,
            "max_acceleration": src["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"],
            "max_jerk": src["robot_cfg"]["kinematics"]["cspace"]["max_jerk"],
        },
    })
    kin.pop("external_asset_path", None)
    kin.pop("external_robot_configs_path", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# GENERATED by gwm_hardware/build_2f140_cfg.py "
              f"--thickness-multiple {THICKNESS_MULTIPLE} -- do not edit by hand.\n"
              "# Panda + Robotiq 2F-140 for this rig. Arm half inherited from\n"
              "# cuTAMP's panda_robotiq_2f_85.yml; gripper spheres fitted to the\n"
              "# 2F-140 collision meshes. See the generator's docstring for the\n"
              "# wrist-camera-sphere caveat.\n")
    out_path.write_text(header + yaml.safe_dump(src, sort_keys=False, width=100))
    return out_path


def main() -> None:
    global THICKNESS_MULTIPLE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-spheres", action=argparse.BooleanOptionalAction,
                    default=True, help="include the placeholder DROID wrist-camera "
                                       "spheres on the gripper base")
    ap.add_argument("--thickness-multiple", type=float, default=THICKNESS_MULTIPLE,
                    help="cap sphere diameter at this multiple of a part's "
                         "thinnest dimension (higher = coarser, fewer spheres, "
                         "faster planning, more over-approximation)")
    ap.add_argument("--out-dir", type=Path, default=ASSETS)
    args = ap.parse_args()

    THICKNESS_MULTIPLE = args.thickness_multiple
    print(f"thickness multiple: {THICKNESS_MULTIPLE}")

    urdf = args.out_dir / URDF_NAME
    if not urdf.exists():
        raise SystemExit(f"{urdf} missing -- run `python -m gwm_hardware.build_2f140` first")
    out = args.out_dir / "panda_robotiq_2f_140.yml"
    print(f"fitting gripper collision spheres from {urdf.name}")
    build_cfg(urdf, out, args.camera_spheres)
    print(f"wrote {out}")
    print("deriving grasp-frame gripper spheres (cuTAMP grasp filter)")
    build_gripper_spheres(urdf, args.out_dir / GRIPPER_SPHERES_NAME)




# ---------------------------------------------------------------------------
# Grasp-frame gripper spheres
# ---------------------------------------------------------------------------
#
# cuTAMP filters candidate grasps by colliding the *gripper* against the target
# object (`particle_initialization.py:154`, via
# `world.robot_container.gripper_spheres`). Those spheres are NOT the per-link
# collision spheres above: they are one flat set in the gripper frame, shipped
# as `robotiq_2f_85_gripper_spheres.pt`, and cuTAMP's docstring warns they are
# "in the origin frame with z-up (not the conventional z-down gripper frame)".
#
# Reverse-engineered from that reference file (23 spheres):
#   x ~ 0            the set is flattened onto the closing plane
#   y = +-0.0621     the CLOSING axis
#   z = 0.002..0.130 the APPROACH axis, 0 at the gripper base
#
# Deriving the same thing from the 2F-85's own URDF + yml (bare gripper, camera
# mount excluded) and searching axis permutations reproduces that envelope only
# under **swap x and y in gripper_frame** -- max envelope error 18 mm, the rest
# being that the reference is hand-authored rather than derived.
#
# So rather than trust an inferred convention, the routine below is calibrated:
# it derives spheres for ANY of these grippers by finding the closing axis
# empirically (the axis the two finger pads separate along) and mapping it to
# y, and `--check-2f85` runs it on the 2F-85 and compares against the shipped
# reference. If it reproduces the 2F-85, it can be trusted on the 2F-140.

GRIPPER_SPHERES_NAME = "panda_robotiq_2f_140_gripper_spheres.pt"
# The reference set is hand-authored, so an exact match is not the bar; this is
# the residual measured when the derivation is run on the 2F-85 itself.
REF_ENVELOPE_TOL_M = 0.020


def _grasp_frame_spheres(robot, spheres_by_link, base_frame, pad_links, driver_cfg,
                         drop_radius=None):
    """Flatten per-link spheres into one set in the gripper frame, closing axis -> y."""
    import numpy as np

    robot.update_cfg(driver_cfg)
    rows = []
    for link, entries in spheres_by_link.items():
        if link.startswith("panda_link") or link == "attached_object":
            continue
        T = robot.get_transform(link, base_frame)
        for e in entries:
            if drop_radius is not None and abs(e["radius"] - drop_radius) < 1e-9:
                continue        # camera-mount rows: absent from the reference set
            c = T @ np.array([*e["center"], 1.0])
            rows.append([*c[:3], e["radius"]])
    pts = np.array(rows)

    # Which axis do the finger pads separate along? That is the closing axis,
    # and the reference convention puts it on y.
    pads = [robot.get_transform(l, base_frame)[:3, 3] for l in pad_links]
    closing = int(np.argmax(np.abs(pads[0] - pads[1])))
    approach = 2
    other = [i for i in range(3) if i not in (closing, approach)][0]
    order = [other, closing, approach]
    out = pts.copy()
    out[:, :3] = pts[:, order]
    return out, order


def build_gripper_spheres(urdf_path: Path, out_path: Path, check_2f85: bool = True):
    import numpy as np
    import torch
    import yaml as _yaml
    from yourdfpy import URDF

    def envelope(p):
        return np.array([[p[:, i].min(), p[:, i].max()] for i in range(3)])

    if check_2f85:
        ref = torch.load(CUTAMP_ASSETS / "robotiq_2f_85_gripper_spheres.pt",
                         map_location="cpu", weights_only=True)
        ref = ref[ref[:, 3] > 0].numpy()
        r85 = URDF.load(str(CUTAMP_ASSETS / "panda_robotiq_2f_85.urdf"),
                        load_meshes=False, build_scene_graph=True)
        cs85 = _yaml.safe_load((CUTAMP_ASSETS / "panda_robotiq_2f_85.yml").read_text())
        got, order = _grasp_frame_spheres(
            r85, cs85["robot_cfg"]["kinematics"]["collision_spheres"], "gripper_frame",
            ["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"],
            {"robotiq_85_left_knuckle_joint": 0.0}, drop_radius=0.025)
        err = float(np.abs(envelope(got) - envelope(ref)).max())
        print(f"  calibration on 2F-85: axis order {order}, "
              f"{len(got)} spheres vs reference {len(ref)}, "
              f"max envelope error {err * 1000:.1f} mm")
        if err > REF_ENVELOPE_TOL_M:
            raise RuntimeError(
                f"the derivation no longer reproduces cuTAMP's 2F-85 gripper "
                f"spheres ({err * 1000:.1f} mm > {REF_ENVELOPE_TOL_M * 1000:.0f} mm). "
                f"Do not trust its 2F-140 output.")

    robot = URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)
    cfg = _yaml.safe_load((urdf_path.parent / "panda_robotiq_2f_140.yml").read_text())
    spheres, order = _grasp_frame_spheres(
        robot, cfg["robot_cfg"]["kinematics"]["collision_spheres"], "robotiq_140_base_link",
        ["left_inner_finger_pad", "right_inner_finger_pad"],
        {DRIVER_JOINT: 0.0}, drop_radius=0.025)
    torch.save(torch.tensor(spheres, dtype=torch.float32), out_path)
    e = envelope(spheres)
    print(f"  2F-140 grasp-frame spheres: {len(spheres)}, axis order {order}")
    for i, ax in enumerate("xyz"):
        print(f"    {ax}: [{e[i, 0]:+.4f}, {e[i, 1]:+.4f}] m")
    print(f"  wrote {out_path}")
    return spheres

if __name__ == "__main__":
    main()
