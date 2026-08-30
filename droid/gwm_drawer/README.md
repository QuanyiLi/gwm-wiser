# gwm_drawer — language-conditioned drawer selection (droid-sim)

Qualitative experiment: three single-drawer cabinets of different colour and
size plus three objects on the table, six candidate trajectories (three
drawer pulls, three object grasps), and GWM asked to select the trajectory
matching a drawer-opening instruction. Each drawer is the target of two
tasks with different referring expressions — its colour ("the red cabinet")
and a size or position attribute ("the largest cabinet", "the middle
cabinet", "the smallest cabinet") — six tasks in all. The object grasps are
distractors: a correct selection has to prefer the right drawer pull over
the other two drawers and over three pick-up motions.

## Scene 8

`make_scene8.py` authors `scene8_0.usd` = stock scene1 (bowl re-placed,
rubiks cube dropped) plus a neutral-gray block, a YCB banana and three
cabinets, each with one drawer in its upper section above a solid base:

- **red** — large, muted red, 30x45x18 cm, knob at z 0.40, at +y
- **yellow** — medium, yellow, 24x35x14 cm, knob at z 0.31, in the middle,
  yawed -25 deg about z
- **blue** — small, blue, 20x27x14 cm, knob at z 0.24, at -y
- objects: block at (0.37, 0.19), bowl at (0.47, 0.05), banana at
  (0.33, -0.09) on the table in front of the cabinets

Carcasses are static colliders; each drawer is one rigid body on a prismatic
joint to the world whose axis is the cabinet's local x (limits
[-pull-1 cm, 0]), spawning with 3 mm reveals and touching nothing (settle
drift 0.0 mm). Both external cameras are re-posed at capture/execution time
(`config.apply_camera_rig`): pos (0.08, ±0.56, 0.44), aimed at
(0.55, ∓0.04, 0.32), focal 2.6 mm — the workspace fills the frame and the
drawers sit near eye level.

## Candidates

`traj.py` builds the six candidates with SAPIEN Pinocchio IK on the scoring
URDF. Drawer pull (in the cabinet's local frame, mapped through its yaw):
home -> stage -> pre-grasp -> knob grasp -> gripper close -> pull along the
slide axis -> hold; the three drawer pulls share one grasp orientation family
and pitch (pads_y+, 25 deg). Object grasp: home -> stage above -> pre above
-> top-down grasp -> close -> 12 cm lift -> hold, at the settled object
position from the capture. Every timeline totals 8.85 s (= the
rat_scale-3.0 RAT window).

## Selection

`run_select.py` scores every candidate under 30 instructions (6 tasks x 5
phrasings) on both external cameras, fuses cameras by the per-candidate mean
and averages each task's phrasings into a 6x6 matrix; the selection for a
task is the argmax of its row.

## Results (ckpt 0810_gwm step 34000)

Ensembled fused matrix (rows = tasks, columns = candidates):

| task | red | yellow | blue | grasp block | grasp bowl | grasp banana |
|---|---|---|---|---|---|---|
| "open the drawer of the red cabinet" | **+0.7055** | +0.6905 | +0.6588 | +0.6156 | +0.6191 | +0.5859 |
| "open the drawer of the largest cabinet" | **+0.6951** | +0.6791 | +0.6457 | +0.5794 | +0.5635 | +0.5587 |
| "open the drawer of the yellow cabinet" | +0.6707 | **+0.7229** | +0.6694 | +0.5923 | +0.5777 | +0.6198 |
| "open the drawer of the middle cabinet" | +0.6686 | **+0.7049** | +0.6635 | +0.5823 | +0.5728 | +0.5849 |
| "open the drawer of the blue cabinet" | +0.6712 | +0.6953 | **+0.6954** | +0.5878 | +0.5876 | +0.5781 |
| "open the drawer of the smallest cabinet" | +0.6746 | +0.7051 | +0.6836 | +0.6013 | +0.5978 | +0.5698 |

- argmax **5/6** correct; margins +0.0150 (red), +0.0160 (largest), +0.0522
  (yellow), +0.0363 (middle), +0.0001 (blue); the miss is "smallest cabinet"
  -> yellow, by 0.0215
- all six tasks pick a drawer pull over the three object grasps; the grasp
  candidates score 0.06-0.13 below the best drawer in every row
- single camera: external_cam_2 argmax 5/6 (same miss), external_cam 4/6
  (both blue tasks -> yellow)

## Execution

`execute8.py` replays the six candidates in Isaac (absolute joint targets at
15 Hz, binary gripper): the drawer pulls open their drawer 12.7 / 8.7 /
7.6 cm (red / yellow / blue = 91 / 87 / 85 % of the pull) along its slide
axis; the grasps lift the block 11.9 cm, the banana 11.7 cm and the bowl
6.4 cm (99 / 98 / 54 % of the 12 cm lift); in every episode every other
drawer and object stays at 0.0 cm. Videos from both external cameras in
`results/exec/`, figures in `results/`.

## Reproduce

```bash
# 1. scene (repo venv has usd-core)
/root/code/gwm/gwm-wiser/.venv/bin/python make_scene8.py
# 2. capture (Isaac)
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u capture8.py
# 3. candidates
/root/code/gwm/gwm-wiser/.venv/bin/python traj.py
# 4. gwm-server (needs the GPU; do not run Isaac concurrently)
cd /root/code/gwm/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
  --backend gwm --urdf droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf \
  --ckpt /root/exp_ret/0810_gwm/checkpoint.pt &
# 5. selection + figures
/root/code/gwm/gwm-wiser/.venv/bin/python run_select.py --dump
../droid-sim-evals/.venv/bin/python plot_figs.py
# 6. execution videos (kill the server first)
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u execute8.py
```

Generated data (`captures/`, `results/`, `scene8_0.usd` — the USD embeds
absolute paths) is gitignored; scenes regenerate from `make_scene8.py`.
