# gwm_push_cem — does a searched trajectory actually push the cube it names?

Three coloured cubes sit around a closed gripper, one in front of it and one to
each side. For each of three prompts — "push the red cube", "push the green
cube", "push the blue cube" — a CEM search over the **endpoint** of a straight
fingertip slide picks a trajectory, and that trajectory is executed in Isaac.

The question is not whether the score picks the right cube. It is whether the
trajectory it picks *moves* that cube: a search that merely parks the gripper
beside the named cube scores just as well and pushes nothing. So the measurement
is physical — where each cube ended up over 100 independent CEM runs per prompt,
300 executed rollouts in total.

## Scene

Robot base at the origin, +x away from the base, +y to the robot's left. The
table top is at z = 0.045141.

| | |
|---|---|
| gripper home | (0.46, 0.00), fingertip 1.5 cm above the table, gripper closed |
| cubes | 47 mm, 25 g, friction 0.45, 20 cm from the gripper |
| red | (0.66, 0.00) — in front of the gripper |
| green | (0.46, +0.20) — to the robot's left |
| blue | (0.46, −0.20) — to the robot's right |

The prompts name the cubes by colour.

## Candidate trajectory

The closed fingertip slides in a straight line at constant height from the home
position to an endpoint (x, y). Duration 4 s, 31 waypoints, interpolated in
Cartesian space with per-waypoint IK from a fixed seed, so the fingertip height
and the tool orientation are the same at every instant.

Only the endpoint is searched. Height and tool orientation are frozen at their
home values, so a candidate is fully described by two numbers.

## Search

Endpoints live on a 2 cm lattice inside a square of half-width 24 cm centred on
the home position; 571 of the 625 lattice points have a fully feasible slide,
the rest being behind the robot.

Scoring goes through the GWM server against both external cameras, fused by the
mean. Every (endpoint, instruction) pair is scored at most once and cached, so
the score map and the search share a single pass.

100 independent CEM runs per prompt, differing only in seed, with the same 100
seeds for all three prompts. Each run: population 24, 4 iterations, elite 6,
σ₀ = 0.10 m, σ floor = one lattice cell, mean initialised at the home position,
samples clipped to the region and snapped to the lattice. Both standard ways of
reading off "the trajectory CEM found" are recorded for every run:

- **winner** — the best-scoring sample the run saw, so the executed trajectory
  is always one the search actually rated;
- **sample** — one draw from the run's converged Gaussian, which takes CEM's
  answer as the distribution it is.

The objective is `lang`: the fused score of a candidate minus the score of the
same candidate under an empty instruction, which removes the part of the score
that any trajectory would earn regardless of what was asked. `--objective raw`
uses the fused score directly.

## Execution

Isaac Lab at 15 Hz, gripper commanded closed throughout, then held still while
the cubes come to rest. Cube spawn and final positions are read from sim state,
not from video.

One episode per *distinct* endpoint rather than per rollout: independent CEM runs
often converge on the same endpoint, and `reset_scene_to_default` puts every body
back bit-exactly, so a repeated endpoint reproduces its own episode. `--verify N`
re-runs N episodes and reports the spread; over the five re-runs in
`results/exec_repeatability.json` the largest is 3e-05 m.

Isaac and the ~19 GB scoring server do not both fit on a 24 GB card, so the order
is: score, stop the server, execute.

## What is measured

Outcomes are bimodal, so there are two thresholds. The closed hand is 2.4 cm wide
at the fingertips, but its knuckles sit at table + 4.7 cm — exactly the height of
a cube top — and are much wider, so a sweep passing several centimetres clear can
still clip a cube and nudge it about a centimetre.

- **moved ≥ 1 cm** — touched at all, knuckle grazes included;
- **pushed ≥ 3 cm** — carried by the closed blade, which is what the prompt asked
  for. This is the reported push rate.

Direction is the angle between the cube's displacement and the direction the
prompt implies, counted correct within 45°.

## Results

Read-off `winner`, 100 CEM runs per prompt:

| prompt | pushed ≥ 3 cm | touched ≥ 1 cm | mean over all 100 | mean push, when it pushed | largest push | distinct endpoints |
|---|---|---|---|---|---|---|
| "push the red cube" | **47/100** | 47/100 | +2.1 cm | 4.5 cm | 7.5 cm | 21 |
| "push the green cube" | **64/100** | 66/100 | +3.1 cm | 4.7 cm | 8.4 cm | 26 |
| "push the blue cube" | **42/100** | 58/100 | +2.8 cm | 5.9 cm | 8.4 cm | 26 |

"mean over all 100" is the displacement along the direction the prompt implies,
averaged over every rollout including the ones that missed.

Three things hold across all 300 rollouts:

- **Every push went the right way.** 153 of 153 pushes landed within 45° of the
  asked direction; the median angular error is 3.6° / 4.4° / 5.3°. There is no
  rollout in which the named cube moved backwards or sideways.
- **No cube other than the named one was ever disturbed** — 0 of 300 rollouts,
  maximum displacement of a non-named cube 0.0 cm. The search is not sweeping
  the table; it is going to one cube.
- **The score map identifies the right cube 3/3, and does so on each camera
  separately** as well as after fusing them, with margins of +0.030 / +0.047 /
  +0.037 fused.

Two things are worth knowing about what the search does *not* do:

- **The blue cube is the one that grazes.** It is touched 58 times but carried
  only 42, against 47/47 for red — its endpoints sit slightly off the line
  through the cube, so the knuckles clip it where the blade would have carried it.
- **The score map's global argmax is not where the search ends up, and that is
  why the search works.** For the red prompt the argmax sits at (0.70, 0.18),
  18 cm from the red cube, and a slide to it would miss entirely. CEM starts at
  the home position and converges to a nearer local optimum at (0.665, 0.081),
  which does reach the cube. Read-off matters for the same reason: taking a draw
  from the converged Gaussian instead of its best sample gives 19 / 46 / 33
  instead of 47 / 64 / 42.

![where the cubes end up](results/fig_cube_final_winner.png)

`results/fig_cube_zoom_winner.png` is the same data per prompt at cube scale,
`results/fig_endpoints_winner.png` shows the endpoints the search chose, and
`results/fig_scoremap.png` is the objective over the whole lattice.

## Reproduce

```bash
cd /root/code/gwm/gwm-wiser/droid/gwm_push_cem
REPO=/root/code/gwm/gwm-wiser

# 1. author the scene and solve the home configuration
$REPO/.venv/bin/python make_scene.py
$REPO/.venv/bin/python validate_setup.py

# 2. capture the frame the scorer conditions on (Isaac venv)
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u capture.py

# 3. start the scoring server in another shell
cd $REPO && .venv/bin/python -m droid.server.gwm_server --backend gwm \
    --urdf droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf \
    --ckpt /root/exp_ret/0810_gwm/checkpoint.pt --port 8902

# 4. score the endpoint lattice, then search
$REPO/.venv/bin/python run_grid.py
$REPO/.venv/bin/python run_cem.py

# 5. stop the server, then execute both read-off rules in one Isaac boot
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u execute.py \
    --plans plans_winner.json,plans_sample.json \
    --out   exec_winner.json,exec_sample.json --verify 5

# 6. numbers and figures
$REPO/.venv/bin/python analyze.py --exec exec_winner.json --out summary_winner.json
$REPO/.venv/bin/python analyze_score_map.py --objective lang
../droid-sim-evals/.venv/bin/python plot_figs.py --exec exec_winner.json --suffix _winner
```

`results/cache_main.json` holds the 571 × 3 scored endpoints, so step 4
costs nothing on a re-run; delete it to score from scratch (~30 min). Step 5 is
then the only slow step, about 9.6 s per distinct endpoint.

Note that the renderer is not bit-deterministic: recapturing the same scene moves
pixels by up to 6/255. That is far below anything visible, but it means a fresh
capture followed by a fresh scoring pass can shift an argmax by a lattice cell.
`results/cache_main.json` is what the reported scores were read from; keep it to
reproduce them exactly.

## Files

| | |
|---|---|
| `config.py` | every constant: geometry, prompts, search region, timings |
| `pushing.py` | IK, straight-line candidates, cached scoring |
| `sim_common.py` | the home-configuration handover the Isaac-side scripts read |
| `make_scene.py` | authors `scene9_0.usd` and symlinks it into the sim assets |
| `validate_setup.py` | solves the home configuration → `results/home_q.json` |
| `capture.py` | one Isaac boot → the RGB/H5 the scorer conditions on |
| `run_grid.py` | scores the whole endpoint lattice → `results/grid.json` |
| `run_cem.py` | 100 CEM runs per prompt → `results/plans_{winner,sample}.json` |
| `execute.py` | runs every rollout in Isaac → `results/exec_*.json` |
| `analyze.py` | the push rates and displacements → `results/summary_*.json` |
| `analyze_score_map.py` | what the score map knows before any search is run |
| `plot_figs.py` | the figures |

## Footprint

Nothing outside this folder is modified except the
`droid-sim-evals/assets/scene9_0.usd` symlink the scene loader's filename
convention requires; the home pose is applied by overriding
`env_cfg.scene.robot.init_state` in memory. Generated data (`captures/`,
`results/`, `scene9_0.usd` — the USD embeds absolute paths) is gitignored; the
scene regenerates from `make_scene.py`.
