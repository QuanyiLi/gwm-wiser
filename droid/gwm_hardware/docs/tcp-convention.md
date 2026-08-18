# Where the TCP goes, and why it is 212 mm

Settled 2026-08-18 after the RC-machine agent proposed 180 mm. Both numbers are
defensible measurements of a Robotiq 2F-140; they measure different things, and
only one of them is the one this pipeline needs.

## The measurements

All from the gripper's base link, so the two grippers are on the same ruler.
Fingertip planes are taken from **collision-mesh vertices**, not link origins —
on both grippers the `*_finger_tip` / `*_inner_finger_pad` origins sit well
short of the physical tip.

| | 2F-85 (what the pipeline was tuned on) | 2F-140 (this rig) |
|---|---|---|
| fingertip plane, **open** | 149.3 mm | **212.0 mm** |
| fingertip plane, closed | 162.8 mm | 235.7 mm |
| pad-origin midpoint, open | 98.3 mm | **177.0 mm** |
| pad-origin midpoint, closed | 111.8 mm | 200.7 mm |
| pad separation, open → closed | — | 136.0 → 7.3 mm |
| **`grasp_frame`** (what the planner commands) | **150.0 mm** | **212.0 mm** |

cuTAMP's 2F-85 puts `grasp_frame` at 150.0 mm and its open fingertip plane is
149.3 mm — the convention is **the fingertip plane with the gripper open**,
to 0.7 mm. 212.0 mm is that same rule applied to the 2F-140.

The proposed 180 mm is close to the 2F-140's **pad-origin midpoint** (177.0 mm),
which is a perfectly sensible TCP for a human driving the gripper by hand. It is
not the quantity this pipeline consumes.

## Why the convention is forced, not chosen

`tiptop/perception/m2t2.py`:

```python
def m2t2_to_tiptop_transform():
    base_to_tcp[2, 3] = 0.1034   # Panda offset
```

M2T2 emits grasp poses in its own convention; tiptop pushes them 103.4 mm along
z to land on `grasp_frame`. **That constant is gripper-agnostic** — it does not
change when the hardware changes. It converts M2T2's pose into the point where
the fingertips should meet the object.

So `grasp_frame` must sit on the fingertip contact plane. Putting it at 180 mm
would make every planned grasp land **32 mm short** of what M2T2 asked for.

## The Franka Desk TCP is a different, unused number

Two facts, both checked in the code rather than assumed:

1. **tiptop never reads Bamboo's `ee_pose`.** It commands joint trajectories
   (`execute_joint_impedance_path`), so Desk's TCP does not affect any executed
   motion. Grep for `["ee_pose"]` outside `kinematics.get_state` returns only
   `gwm_tiptop/grasp_gate.py` and `place_propose.py`, both of which use cuRobo
   FK, not the robot's report.
2. **Hand-eye calibration uses cuRobo's own FK.**
   `calibrate_wrist_cam.py:557` is
   `motion_gen.kinematics.get_state(get_q_curr()).ee_pose`; the line that would
   have read the robot's `ee_pose` (`:602`) is commented out.

Consequence: the camera extrinsic is solved **relative to `grasp_frame`**, so
the calibration is self-consistent with whatever value we pick — but it bakes
that value in. **Re-run hand-eye calibration if `grasp_frame` ever changes.**

Measured on the live robot (6 samples, flange→EE recovered by FK from reported
`qpos` and `ee_pose`):

```
mean = [+0.00, -0.00, +212.00] mm    spread 0.00 mm    rotation 0.00 deg
```

Desk already stores 212.0 mm. Our URDF has 211.963 mm, so the agreement is two
independent numbers landing 0.037 mm apart, not a circular check. **Leave Desk
at 212** — not because tiptop needs it, but so both machines report the same
end-effector position and a disagreement means something is actually wrong.

Separately: Desk's **payload** fields (mass, centre of mass, inertia tensor)
are *not* the TCP and *do* matter. Wrong payload shows up as reflex stops under
load — i.e. exactly when the gripper is holding something, not during the
free-space moves that pass so reassuringly.

## The one behavioural difference to watch

Because `grasp_frame` is defined with the gripper **open**, the fingertips
advance past it as the gripper closes:

| | tip advance, open → closed |
|---|---|
| 2F-85 | 13.5 mm |
| 2F-140 | **23.7 mm** |

The 2F-140 scoops nearly twice as far. This is the adaptive linkage doing what
it is designed to do, not a modelling error, and with an object present the
fingers stop on contact — so 23.7 mm is an upper bound in free air, not a fixed
offset. But `grasp_gate.py`'s thresholds (`MIN_SLAB_PTS`, `MIN_THICKNESS`, the
pad capture box) were calibrated against the 2F-85's geometry and will need
re-measuring on this gripper once there are real grasp outcomes to tune on.


## Addendum 2026-08-18 — the orientation half of the convention

`grasp_frame`'s **rotation** is as constrained as its translation, and for the
same reason: `m2t2_to_tiptop_transform()` is gripper-agnostic, so the frame has
to match the 2F-85 the pipeline was tuned on.

The first generated 2F-140 got this wrong by 90 deg about the approach axis:

```
2F-85            x=[0,1,0]   y=[-1,0,0]  z=[0,0,1]
2F-140 (broken)  x=[-1,0,0]  y=[0,-1,0]  z=[0,0,1]
```

Fixed with a -90 deg yaw on `gripper_joint` in `build_2f140.py`.

Why it is worth a section of its own: **the closing axes coincide at equal
joint angles**, so the two models look identical when compared link by link,
and all nine of the then-existing validation checks passed. The error only
materialises when the planner is asked to satisfy a grasp pose — it rotates the
wrist 90 deg and closes across the wrong axis of the object.
`validate_2f140.py` now compares the two conventions directly (check 2b).

Found because the controller-machine agent hit the same class of bug from the
other end: the stock Franka ready pose carries `q7 = +pi/4` to cancel the
Franka Hand's -45 deg flange mounting offset, and once a Robotiq is bolted on
square, the compensation *is* the error. Our poses were already clear of it
(`q_home` and cuTAMP's neutral pose both use q7 = 0), but the prompt to go
looking is what surfaced the 90 deg.

## Addendum 2026-08-18 — hand-eye calibration, and what it replaced

`calibrate-wrist-cam` ran against this rig's own board (11x8, 34.31 mm checker,
DICT_5X5_100 — see below). Result for camera `035422072950`, as
`ee_from_cam` in **grasp_frame**:

```
translation  [-36.76, -65.39, -146.09] mm
rotation     [-0.93, -0.41, -0.33] deg
```

Every quantity that had been estimated now has a measured replacement:

| | estimated | measured | |
|---|---|---|---|
| camera behind the TCP | 136 mm | **146.1 mm** | from a single table-plane fit |
| optical axis vs approach | 2.75 deg | **1.01 deg** | plane-fit error dominated the estimate |
| lateral offset | unknown | **75.0 mm** | this is what kept the framing off |
| bracket side | unknown (spheres mirrored) | **+y** | `[-65.4, +36.8, +65.9]` mm in the gripper base frame |

Consequences applied:

- **Camera collision spheres now come from the calibration**
  (`build_2f140_cfg.camera_mount_spheres`), enclosing a 90 x 25 x 25 mm D435
  body swept along the measured optical axis plus a bracket run back to the
  gripper base. They were previously mirrored to both sides as a hedge.
- **The DROID ZED bracket links were dropped from the URDF.** In grasp_frame
  they sat 318 mm and 279 mm off the approach axis; the real camera is at
  75 mm. Meshes wrong by 240 mm are worse than no meshes.

### Board parameters

The board is not DROID's. Identified from a photo with the aruco detector
rather than by counting squares: **11 x 8, DICT_5X5_100**, 44 markers
(ids 0..43), all 70 interior corners recovered.

Its checker size was measured **with the depth camera**, because the tape
readings disagreed — 37.5 cm across 11 squares implies 34.09 mm, 27.5 cm across
8 implies 34.38 mm, and 0.8 % of scale error lands directly in the hand-eye
solve. Unprojecting the detected corners with metric depth and measuring
neighbour distances gives:

```
FoundationStereo   34.31 mm   spread 0.04 mm over 5 shots
RealSense ASIC     34.32 mm   spread 0.05 mm over 5 shots
```

Two independent depth sources agreeing to 0.01 mm. 34.31 mm it is — which makes
the grid 37.7 x 27.4 cm, so the 27.5 cm reading was right and the 37.5 cm one
was 2 mm short. Installed by `install_charuco_params.py`; the upstream 14x9 /
20 mm block is kept as `calibrate_wrist_cam.py.orig`.

### q_capture, validated against the calibration

`q_capture` was chosen empirically (sweep, photograph, score) before the
extrinsic was known. Re-deriving its footprint from the calibrated camera pose
confirms the choice:

```
camera at [+0.515, +0.037, +0.696] m, 641 mm above the table
footprint x [0.250, 0.756]   y [-0.408, +0.487]
covers M2T2's y in [-0.30, +0.30] crop:  YES
robot base plate (x < 0.25) in view:     no
```

The near edge lands at x = 0.250, and 0.25 is also where the base plate was
judged to end when reading it off the first capture images — two independent
routes to the same number. The footprint is off-centre in y (+0.04 rather than
0), which is exactly the 75 mm lateral camera offset showing through.
