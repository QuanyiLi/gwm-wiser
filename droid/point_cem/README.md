# point_cem — 4-image pointing: GWM score maps + CEM planning

*(standalone experiment folder; nothing outside this folder is modified
except the `scene7_0.usd` symlink the scene loader requires)*

For a prompt naming one of four photos on the table, the GWM score over EEF
hover positions is highest over that photo, and CEM over the EEF's (x, y)
goal — sampling actions and scoring them with the model, as in LeWM-style
planning — converges onto it.

## Scene

`scene7_0.usd` = stock scene1 (table + robot + dome) minus its objects, plus
four 15 cm photo quads in a 2×2 (0.20 m pitch), inside the reach band and
both external cameras' views. The gripper is closed at capture and stays
closed throughout.

| cell | image | world (x, y) |
|---|---|---|
| near-left | dog | (0.37, 0.00) |
| far-left | panda | (0.57, 0.00) |
| near-right | banana | (0.37, −0.20) |
| far-right | strawberry | (0.57, −0.20) |

Two animals + two fruits; "the animal" / "the fruit" are 2-way ambiguous
control prompts.

## Candidates

A candidate = hover the closed gripper's fingertip at (x, y, table + 5 cm),
tool axis straight down, yaw fixed at the home yaw. Joint trajectory =
home → IK target, linear in joint space, uniform 6 s / 31 waypoints, gripper
closed throughout. IK = SAPIEN Pinocchio on the scoring URDF
(d_tip = 0.1625 m).

## Scoring

Unmodified droid-sim path: `gwm-server` (fp32 head, ckpt 0810_gwm step
34000), RAT = [scene photo, 5 robot-only renders], rat_scale 3.0, task_image
current. Both external cameras, fused by the mean; the per-candidate prior
(empty-instruction cosine) is recorded for every point. Prompt template:
`point at the image of the {name}`.

## Runs

- **Grid**: 19×20 = 380 hover points (2 cm), 6 prompts (4 specific + animal
  + fruit), 2 cams → the score maps (grid.json, stats.json).
- **CEM**: one run per specific prompt; pop 30, 5 iters, elite 8,
  σ₀ = 10 cm isotropic at the region centre, σ floor 8 mm, samples snapped
  to a 1 cm lattice (cache shared with the grid). Objective = fused score
  (cem.json).
- **CEM step frames**: for each task, the arm driven to every iteration's
  mean and then to the selected point, with both external cameras saved
  (results/cem_frames/, via prep_snap.py + snap_cem_poses.py).
- **Bar-EEF ablation**: same candidates, second gwm-server whose renderer
  URDF replaces the 2F-85 with a rigid bar of the same tip offset
  (grid_bar.json, stats_bar.json).

## Results

Fused score over the 380-point grid (stats.json; "cell margin" = mean score
over the prompt's cell minus the best other cell's mean):

| prompt | argmax in cell | top-10 in cell | cell margin | CEM hit (mean / best sample) |
|---|---|---|---|---|
| dog | ✓ | 0.90 | +0.0041 | ✓ / ✓ |
| panda | ✓ | 0.80 | +0.0045 | ✓ / ✓ |
| banana | ✓ | 0.90 | +0.0107 | ✓ / ✓ |
| strawberry | ✓ | 0.90 | +0.0124 | ✗ / ✓ |

- Grid argmax in the correct cell 4/4; every cell margin positive.
- CEM (pop 30 × 5 iters, one seed per prompt): the selected point (the
  run's best-scoring sample) is in-cell 4/4; the sampling distribution's
  final mean is in-cell 3/4 (strawberry's mean stops on the flat high-score
  region at the table centre, visible in fig_maps_perprompt).
- Cameras: fused margins are positive on all four prompts; each camera alone
  has prompts with negative margins (per-camera tables in stats.json).
- Controls: the empty-instruction prior map is diffuse; "the animal" /
  "the fruit" highlight the union of their two referents; the z-scored
  cross-prompt partition assigns each photo's quadrant to its own prompt.
- Bar-EEF ablation (stats_bar.json): argmax in cell 3/4, cell margins
  dog −0.0024 / panda +0.0033 / banana +0.0029 / strawberry −0.0132.

Figures: `fig_cem_combined.png` (CEM clusters per iteration, all prompts),
`fig_maps_perprompt.png` (+`_debiased`), `fig_controls.png`,
`fig_argmax_partition.png` (+`_debiased`), `fig_bar_compare.png`;
per-candidate scorer inputs in `strips_gripper/`, `strips_bar/`;
arm-at-each-CEM-step camera frames in `cem_frames/{task}/`.

## Repro

`scene7_0.usd` is generated, not committed — it embeds machine-local absolute
paths (payloads + textures). `make_scene7.py` rebuilds it on each checkout
from the stock `scene1_0.usd` plus `assets/img/`, and creates the
`droid-sim-evals/assets/` symlink itself.

```bash
# 1. scene + capture (Isaac; GPU must be free of the server)
.venv/bin/python make_scene7.py
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u capture7.py
.venv/bin/python validate_setup.py

# 2. scoring server (repo venv, from the repo root)
.venv/bin/python -m droid.server.gwm_server --backend gwm \
  --urdf droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf \
  --ckpt /root/exp_ret/0810_gwm/checkpoint.pt --port 8901

# 3. experiment (repo venv)
.venv/bin/python run_grid.py
.venv/bin/python run_cem.py --only dog,panda,banana,strawberry
.venv/bin/python analyze.py
.venv/bin/python dump_strips.py --out strips_gripper

# 4. CEM step frames (Isaac; stop the server first)
.venv/bin/python prep_snap.py
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u snap_cem_poses.py

# 5. bar ablation: restart the server with --urdf droid/point_cem/assets/panda_bar.urdf, then
.venv/bin/python run_grid.py --only dog,panda,banana,strawberry --cache cache_bar.json --out grid_bar.json
.venv/bin/python dump_strips.py --out strips_bar

# 6. figures (droid-sim venv has matplotlib)
../droid-sim-evals/.venv/bin/python plot_figs.py --bar-grid grid_bar.json
```

## Notes

- Scores are ranked, never thresholded; maps use per-panel relative ramps.
- Cross-prompt comparisons z-score each prompt's map first.
- CEM optimises a goal pose scored by the model's semantic-outcome
  prediction: the LeWM action-sampling loop with a full-reach score rather
  than a receding-horizon rollout.
