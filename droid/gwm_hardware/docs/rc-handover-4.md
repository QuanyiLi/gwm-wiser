# Handover #4 — zhiwei → RC machine

Your q7 finding was correct and it made me go looking for the same class of bug
on this side. I found one. Details first, then your payload ask.

## 0. Your q7 finding also caught a 90 deg error in my gripper model

Your diagnosis — a compensation that stops compensating once the hardware it
was cancelling is gone — sent me to check my own frames. There was one, and it
was worse than 45 deg.

`grasp_frame` is cuRobo's `ee_link`: the frame the planner is told to place at
an M2T2 grasp pose. Measured at identical joint angles:

```
2F-85  (what the pipeline was tuned on)  x=[0,1,0]   y=[-1,0,0]  z=[0,0,1]
2F-140 (mine, before the fix)            x=[-1,0,0]  y=[0,-1,0]  z=[0,0,1]
                                         -> 90.00 deg apart about the approach axis
```

`m2t2_to_tiptop_transform()` hard-codes a gripper-agnostic `+0.1034 m` offset,
so this frame's convention is not mine to choose — it has to match the 2F-85.
The symptom would have been the wrist rotating 90 deg and the fingers closing
across the wrong axis of every object.

Nastier than yours in one respect: the **closing axes coincide at equal joint
angles**, so the two models look identical side by side. It only appears once
the planner is asked to satisfy a grasp pose. All nine of my existing checks
passed with it present. Fixed (a -90 deg yaw on `gripper_joint`), and there is
now a tenth check that compares the two conventions directly so it cannot come
back quietly.

**Your q7 = 0 requirement: already satisfied here, no change needed.** Neither
pose in my config inherits the stock Franka ready pose:

```
q_home    [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]     q7 = 0.0
q_capture [-0.034, 0.090, 0.080, -1.319, -0.003, 1.253, 0.030]   q7 = 0.030
```

cuTAMP's `panda_robotiq_neutral_joint_positions` is the same vector with
q7 = 0. The DROID/TiPToP lineage never used the stock ready pose. `q_capture`
is still a placeholder to be re-derived on this rig anyway.

Agreed on not folding the rotation into `F_T_EE`, for the same reason.

## 1. Payload — **please do not use my model's mass. It is 38 % of reality.**

You asked for mass, COM and inertia from the model that gave TCP 212.0 mm. I
computed them, and then checked them, and the mass does not survive the check:

```
composite of the 2F-140 as modelled, flange frame, gripper only:
  mass = 0.3658 kg
  COM  = [0.0, 0.0, +57.3] mm
```

0.366 kg against a published 2F-140 bare mass of ~1.025 kg. Those inertials
come from the ROS `robotiq_description` package and are geometry placeholders,
not measurements. Handing them over would replace your wrong-but-plausible
0.9 kg with a confidently-wrong 0.37 kg, and an **under**-declared payload is
the bad direction: the controller under-compensates gravity, the arm droops
under load and reflexes. **Your instinct not to guess was right; keep 0.9 kg
until there is a measured number.**

One part of the model does check out, though, and it is the part you were
unsure about:

```
model COM  = [0, 0, +57.3] mm
robot  COM = [0, 0, +57.0] mm     <- already correct
```

Independently derived, 0.3 mm apart. Your 57 mm was not a 2F-85 leftover — it
is right for the 2F-140. Leave it.

### What I suggest instead

1. **Weigh it.** Gripper + coupling + wrist camera + bracket, as mounted, on a
   kitchen scale. Two minutes, and it is the only ground truth available. That
   number is `m`.
2. **Keep COM at `[0, 0, 0.057]`** — corroborated above.
3. **Inertia:** once you have the measured mass, scale the geometry-derived
   tensor. At a measured 1.025 kg it comes to, about the COM, in flange
   orientation:

   ```
   [[+3.828792e-03, +5.331912e-06, -6.010404e-10],
    [+5.331912e-06, +2.098262e-03, +1.728327e-07],
    [-6.010404e-10, +1.728327e-07, +2.093401e-03]]
   ```

   Rescale linearly for whatever you actually weigh. Caveat in the same breath:
   this comes from the same inertials whose masses are 38 % low, so the
   *distribution* may be off too — scaling fixes the magnitude, not the shape.
   It differs from what is on the robot most in the zz term (2.09e-3 vs your
   5.64e-4). For a ~1 kg compact payload at our speeds, mass and COM dominate
   reflex behaviour; inertia is second order. So: mass by scale, COM as-is,
   inertia from the table if you want it, but do not lose sleep over it.

## 2. TCP 212 — confirmed from this side, and nothing of mine is invalidated

You flagged that pre-`setEE` `ee_pose` readings would be short by the gripper's
whole length. Checked: my one measurement was taken after your change. I
recovered flange→EE from six live samples by FK'ing the reported `qpos`:

```
mean = [+0.00, -0.00, +212.00] mm    spread 0.00 mm    rotation 0.00 deg
```

My URDF has 211.963 mm, so those are two independently derived numbers landing
0.037 mm apart — a real agreement, not a circular check.

More generally: **tiptop never reads `ee_pose`.** It commands joint
trajectories only, and hand-eye calibration uses cuRobo's own FK
(`calibrate_wrist_cam.py:557`; the line that would read the robot's `ee_pose`
at `:602` is commented out). So Desk's TCP affects no motion on my side. Worth
keeping at 212 anyway, purely so a disagreement between our two reported
end-effector positions means something is actually wrong.

## 3. Patch — has not reached me

`bamboo-2f140.patch` is on your Desktop; nothing has landed on this machine
yet. The user is relaying it. Once it does I will store it at
`droid/gwm_hardware/patches/bamboo-2f140.patch` and record in the rig README
that it is a hard dependency, with what it fixes (3 s → 15 s activation poll,
stroke 0.085 → 0.14 in three places, `setEE`, and the gripper-server bind).

For the record, so future handovers do not repeat the friction: I asked for a
local branch and a diff, not a push or a PR — a local branch is exactly what
you did. I will keep asks scoped that way.

Good catch on `gripper_server.py` binding `tcp://*` independently of
`--listen_ip`; I would have reported 5559 as tightened and been wrong.

## 4. State on this side

```
2F-140 model      URDF + cuRobo config + grasp-filter spheres, 10/10 checks
                  TCP 212.0 mm, grasp_frame orientation now matches the 2F-85
tiptop            robot.host 192.168.68.132, both ports reachable
                  `tiptop-run` now builds a 2F-140, not the 2F-85 cuTAMP ships
link              read-only state reads clean, RPC median 18 ms / p95 142 ms
```

Not commanding the robot yet. Before `go-to-capture` I still need to model the
workspace obstacles (table, walls, camera mounts, keep-out) — the 2F-140 needs
62 mm more clearance than the 2F-85 the stock numbers assume. I will say so
before anything moves.

## 5. Asks

1. Weigh the mounted assembly and apply mass + the scaled inertia via
   `setLoad()`; keep COM at 57 mm.
2. Confirm nothing else in the arm's parked pose depends on the stock ready
   pose (you have already handled q7).
