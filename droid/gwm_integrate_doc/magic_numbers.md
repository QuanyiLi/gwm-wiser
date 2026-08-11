# gwm_tiptop Magic-Number Registry

*Census from the oracle/hardcode audit of 2026-08-11 (all 8 package files read line-by-line; executor-mirror constants cross-checked against `droid-sim-evals` and `tiptop` sources). Line numbers are as of that date — anchor by function/constant name if they drift. Companion to [plan.md](plan.md).*

**Audit verdict (context for this list):** no runtime oracle leaks — both proposers consume only `rgb/depth/K/pos_w/quat_w_ros/q_init` plus the robot's own model. Every constant below is either a platform prior, an inherited shared-infra value, a scenario assumption, or a threshold *tuned/validated on the current eval assets*. The registry exists so those last two classes get re-checked before any new scene, asset, or rig.

## Risk classes

| Class | Meaning | Action when scene/assets change |
|---|---|---|
| **A** rig prior | DROID-platform fact (robot bolted to table) | declare in paper; revisit only on a new rig |
| **B** shared infra | inherited from tiptop/M2T2/executor, identical for both eval arms | keep in sync with upstream; declare |
| **C** scenario assumption | encodes "how our scenes are laid out" (objects rest on table, away from edges, …) | re-check against the new layout **first** |
| **D** asset-tuned | mechanism is scene-agnostic but the value was calibrated/validated on the current assets (KLT bins, bowl, banana, 30 mm block) | re-validate on new assets; sensitivity-test before generalization claims |
| **E** policy margin | robot-side clearance/offset choice | re-validate, low risk |
| **F** executor mirror | MUST equal the executing client's value or plans/scoring silently skew | verified equal 2026-08-11; re-verify after any executor change |
| **G** generic numerics | voxel sizes, RANSAC/DBSCAN params, percentiles | low risk, standard tuning |

## Re-check-first shortlist (new scene checklist)

1. **M2T2 workspace crop `x∈[0,1], y∈[−0.3,0.3], z∈[−0.2,0.5]`** — upstream, both arms. Scene-6 banana sits at y=−0.243, ~6 cm from the y-edge. An object outside the box gets **zero grasps and is silently dropped from movables** (no error). `M2T2/m2t2_server.py:50`, enabled via `tiptop/config/tiptop.yml` `apply_bounds: true`.
2. **`resting_tolerance = 0.04` / `FLOAT_TOLERANCE = 0.04`** — clusters whose lowest point is >4 cm above the table are deleted as "robot arm". Assumes every object rests directly on the table: stacked objects or a tall object occluded below 4 cm vanish.
3. **`min_cluster_points = 120`** — small/distant objects vanish below this.
4. **`HOLLOW_MIN_DEPTH = 0.030` + rim `coverage ≥ 0.8`** — container-vs-solid decision, calibrated against the KLT bins/bowl (the "KLT floor within 1 mm of table plane" measurement in `landing_surface` comments). Deep-narrow or low-walled containers unproven.
5. **`xy_margin = 0.02` + table AABB 2/98-percentile crop** — objects overhanging the table edge get clipped out.
6. **`XY_OFFSETS` max ±18 mm** — sized against the 0.105 m bin mouth (self-consistent with the gripper-fit argument: block edge at 33 mm < 52.5 mm half-mouth, but re-derive for any new container/held object).
7. **`z_band = (−0.25, 0.15)`** — table-plane search band, rig prior; wrong on any rig where the robot is not table-mounted.
8. **grasp-gate `MIN_THICKNESS = 0.015`** — vetoes ALL thin-wall pinch grasps (bin walls 4–8 mm, bowl rim 4–7 mm — the latter went 10/10 in G-24, so these are false vetoes). Safe today only because the two-stage `--apply` falls back to the ungated winner inside the *selected* object; re-tune (e.g. a "thin but symmetric and deep" branch) before gating any rim-pinch target.
9. **`score_client` two-stage selection × the per-object quota** — the object choice is only as stable as the number of candidates each object gets. `se3_fps_indices` samples each object's quota by SE(3) *diversity*, so the candidates are the extremes of that object's grasp family, not its mode; reducing by `max` then compares order statistics of extremes across objects at GWM's ~0.01 within-family spread. Measured on scene6_rev2 (16 candidates / 5 clusters / 10 instructions): `max` 6/10 correct objects, `mean` 9/10 (G-28). The same diversity property is why stage 2 (which grasp) uses M2T2 confidence, not GWM. Rule and quota both need re-checking whenever `--k-total` or the cluster count changes — the aggregate's winning margins are +0.005…+0.076, i.e. still small.
10. **`score_client --cam` (the scoring viewpoint)** — the single largest lever measured so far: the SAME candidates and the SAME instructions score 9/10 vs 10/10 correct objects from the rig's two external cameras, with margins differing 5–7× on the tasks whose target is small or gripper-shadowed in one view (G-29). Look at both captures before trusting any selection number on a new layout.

## perception_geometric.py

### `find_table_plane` (signature defaults)

| Constant | Value | Class | Notes |
|---|---|---|---|
| `max_planes` | 5 | G | RANSAC peel iterations |
| `normal_z_min` | 0.90 | G | horizontality: \|n_z\| ≥ 0.90 (≈ ≤26° tilt) |
| `z_band` | (−0.25, 0.15) | **A** | table height band rel. robot base — DROID bolts robot to table |
| `workspace_radius` | 1.4 m | A | XY crop radius around base |
| `voxel_size` | 0.005 m | G | downsample before RANSAC |

Body: min 50 pts to keep peeling (`:54`); RANSAC `distance_threshold=0.01, n=3, iters=1000` (`:56`); outlier removal `(20, 2.0)` (`:80`); plane DBSCAN `eps=3·voxel=0.015, min_points=10` (`:81`); table XY AABB from **2/98 percentiles** (`:88`, class C — edge overhang clipped); box top sunk **0.02 m** below surface (`:96`, class B — tiptop collision convention; the returned `surface_z` is the real boundary).

### `_merge_xy_overlapping_clusters`

| Constant | Value | Class | Notes |
|---|---|---|---|
| containment `frac` | > 0.15 | D | occluded-body merge (bowl seen as two rim arcs) |
| `min_xy_dist` | < 0.008 m | D | rim-sliver merge |

### `cluster_objects`

| Constant | Value | Class | Notes |
|---|---|---|---|
| `eps` / `min_points` | 0.015 / 40 | G | object DBSCAN |
| `min_cluster_points` | 120 | **C/D** | smaller clusters vanish (shortlist #3) |
| `max_objects` | 8 | C | cap on cluster count |
| `xy_margin` | 0.02 m | **C** | crop inside table footprint (shortlist #5) |
| `voxel_size` | 0.004 m | G | |
| `resting_tolerance` | 0.04 m (`:214`) | **C** | floating-cluster = arm filter (shortlist #2) |

Body: per-object outlier removal `(10, 2.0)` (`:234`).

## propose_from_h5.py (pick driver)

| Constant | Value | Class | Notes |
|---|---|---|---|
| extrinsics z correction | −0.015 m (`:57`) | **F** | grasp-frame correction; **verified identical** to `droid-sim-evals/src/sim_evals/inference/tiptop_websocket.py:250` (the client feeding the baseline arm). NB `tiptop_offline.py:205` uses the *opposite sign* — different h5 format, not this eval's path |
| depth validity clamp | (0.05, 4.0] m (`:154`) | G | |
| above-table cut | `surface_z + 0.015` (`:180`) | D | shared with place driver |
| `--k-total` | 16 | E | scene-independent candidate budget (G-5) |
| `--num-particles` / `opt_steps` / `--max-planning-time` | 256 / 500 / 60 s | G | cuTAMP budgets |

## proposals.py

| Constant | Value | Class | Notes |
|---|---|---|---|
| `rot_weight` | 0.1 | G | SE(3) FPS metric: meters + 0.1·radians |
| confidence clip | ≥ 1e-3 | G | |
| `max_refine_per_candidate_slack` | 2 | G | refine up to 2·quota particles |
| *FPS selection policy* | diversity-first | **D** | not a number but the same hazard: `se3_fps_indices` maximizes `(min SE(3) distance) × confidence`, so a small quota returns the grasp family's **extremes** (corner clips, rim pinches — G-26's 0/5 `plan_12` and every bin-wall candidate). Consumers must not treat one object's best candidate as representative of that object; see shortlist #9 |
| StablePlacement tol / mult | 1e-2 / 1.0 | B | mirrors `tiptop.planning.run_planning` loosening |

## place_propose.py

Module-level (named, commented in-file — the already-centralized ones):

| Constant | Value | Class | Notes |
|---|---|---|---|
| `XY_OFFSETS` | 9 offsets, max ±0.018 m | **D/E** | deterministic per-destination pattern (shortlist #6) |
| `APPROACH_CLEARANCE` | 0.055 m | E | held-object bottom above rim on approach |
| `LANDING_CLEARANCE` | 0.010 m | E | hover above landing surface at descent end |
| `HOLLOW_MIN_DEPTH` | 0.030 m | **D** | container-vs-solid (shortlist #4) |
| `HAND_CROP_RADIUS` | 0.20 m | E | in-hand detection crop; also cuts hand region out of the scene cloud |
| `FLOAT_TOLERANCE` | 0.04 m | **C** | same assumption as `resting_tolerance` (shortlist #2) |
| `SELF_SPHERE_PAD` | 0.010 m | E | padding on robot collision spheres before rejecting a point as "self" |

`estimate_held_object`: ceiling `ee_z + 0.12` (`:119`, **inline, unnamed** — worst offender); min 40 pts; DBSCAN `eps=0.008, min_points=25`; bottom/top = 2/98 percentiles.

`landing_surface` (all inline, all class **D** — calibrated against measured KLT/bowl geometry per in-code comments):

| Constant | Value | Notes |
|---|---|---|
| floor-inclusion cut | `table_z − 0.010` | thin-shell floor sits AT plane height (measured ≤1 mm) |
| above-table band | `table_z + 0.015`, ≥60/≥40 pts | footprint occupancy gates |
| `rim_z` | 98th pct of above | |
| boundary sampling | 0.004 m steps | |
| interior erosion | `d_boundary > 0.015` | separates floor from side faces |
| low-point gate | ≥30 pts | |
| rim band | `rim_z − 0.015` | |
| enclosure | 24 angular bins, coverage ≥ 0.8 | rejects crescents (banana, split arc) as containers |
| `land_z` | 10th pct of low pts | |
| landing-xy band | `land_z ± 0.012`, ≥20 pts | aim where floor/top was actually seen |

`main`: depth clamp (0.05, 4.0]; cluster cut `surface_z + 0.015`; solver build `num_particles=32, num_spheres=64`; plan `timeout=2.0 s`; descend `hold_vec_weight=[0.1×5, 0.0]` (B — cuTAMP constrained-descent trick verbatim); close pause 1.33 s (F, see score_client).

## score_client.py

| Constant | Value | Class | Notes |
|---|---|---|---|
| `GRIPPER_PAUSE_S` | 20/15 s ≈ 1.333 | **F** | **verified identical** to executor defaults `gripper_action_steps=20, sim_control_hz=15.0` (`tiptop_websocket.py:41-42`) |
| `GRIPPER_PAUSE_SUBSTEPS` | 7 | G | scoring-side timeline discretization only (no executor counterpart) |
| `--rat-scale` | 3.0 | — | the *real* hyperparameter (G-20 decision), not a magic number; `none` = uniform-6 |
| `--cam` | `external_cam_2` | **D** | which of the rig's two third-person cameras the scene is scored from (G-29). Switched from `external_cam` on 2026-08-11: object accuracy 9/10 → 10/10 and the banana instructions' margins grew 5–7× purely because that view is not gripper-shadowed. **First-order effect, and chosen post-hoc** — re-pick (or implement the unbiased both-cameras average) for any new layout rather than inheriting this one. The camera POSES themselves are upstream/class-B (`droid_environment.py:44`) |
| `--object-score` | `mean` | **D** | stage 1 of selection (G-28): the OBJECT is chosen by this aggregate over its candidates' GWM scores. `max` reproduces the pre-2026-08-11 global per-candidate argmax. Chosen over `median`/`logsumexp`/`top2-mean` (all also 9/10 on scene6_rev2) for size-normalisation — quotas are floor+remainder, so families differ by one — and the best worst-case margin (+0.0052 vs median's +0.0015). See shortlist #9 |
| *within-object rule* | M2T2 confidence | **D** | stage 2 (G-28, user-directed): WHICH grasp of the chosen object, GWM score only as tie-break. GWM cannot see grasp robustness (robot-only RAT frames), and on scene6_rev2's cube its within-object argmax is the 0/5 corner clip `plan_12` (conf 0.341) while M2T2 ranks the 5/5 `plan_10` first (conf 0.778). Validated against the closing-line gate on all 5 objects of one pool + one execution ground truth — **not** an established correlation; re-check when execution data accumulates. cuTAMP soft cost was considered and rejected as the stage-2 signal: `get_ranked_satisfying_particles` already prefers confidence over cost, the cost is never returned, and it scores constraint satisfaction (reach/collision), not grasp stability |
| request timeout | 1800 s | G | |

## grasp_gate.py (closing-line grasp gate, added 2026-08-11 after the census — G-27)

| Constant | Value | Class | Notes |
|---|---|---|---|
| `MIN_SLAB_PTS` | 150 | **D** | ABSOLUTE raw-point count in the capture box — scales with camera resolution × working distance. Recalibrate on a new rig, or replace with a per-cluster fraction |
| `MIN_THICKNESS` | 0.015 m | **D** | solid-body prior (shortlist #8): false-vetoes thin-wall rim pinches that physically work |
| `MAX_CENTER_OFF` | 0.015 m | E | slab asymmetry between pads (one-pad-first shove precursor) |
| `MAX_ORTHO_OFF` | 0.012 m | E | slab offset across the pad width (sideways edge clip) |
| `BOX_PAD` | 0.005 m | E | slop on the self-calibrated pad window |
| `HAND_CROP_RADIUS` | 0.20 m | E | same value/purpose as place_propose (capture-pose gripper crop) |
| tip-band filter | z_ee > −0.06, r < 0.03 | **B** | robot-model facts: this URDF's TCP sits at the fingertip plane, arm spheres at −z, pads = small spheres near TCP; cuRobo pads the sphere buffer with negative radii (filter `r > 0` first). Re-derive the band for any other gripper/URDF |
| open-gripper sanity | `open_half ≥ 0.02` | G | calibration guard (gripper must be open in the model) |

Validation breadth: scene6_rev2 pool + nearbowl only (fragile plan_12 n=62/center 30 mm FAIL vs robust plan_10 n=5718/thick 64 mm/center 0.6 mm PASS; execution 0/5 → 5/5). Cross-tag regression over all refer6 instructions not yet run.

## policy_server.py

| Constant | Value | Class | Notes |
|---|---|---|---|
| q_init drift threshold | 1e-3 rad (`:84`) | C | **warn-only** — a scene that no longer matches the capture still gets served the stale plan. Consider a `--strict` fail mode |

## validate_renderer_overlay.py (debug tool)

`GWM_WISER_ROOT = /root/code/gwm/gwm-wiser` hardcoded absolute path (`:22`); robot mask threshold `max > 8`; blend 0.45/0.55. `assets/panda_robotiq_droidsim.urdf` carries the 18.2 mm Robotiq standoff read from the sim USD — robot *self*-model calibration, not scene knowledge (real-world analogue: measuring the real gripper).

## Upstream / inherited (outside this package — a constants.py could never own these)

| Constant | Value | Where | Class |
|---|---|---|---|
| M2T2 workspace crop | x [0,1], y [−0.3,0.3], z [−0.2,0.5] | `M2T2/m2t2_server.py:50`; `tiptop.yml apply_bounds: true`; used identically by baseline (`tiptop/perception_wrapper.py:110`) | **B/C** (shortlist #1) |
| `contact_threshold_m` | 0.01 | `tiptop.yml` | B |
| `voxel_downsample_size` | 0.0075 | `tiptop.yml` (M2T2 input) | B |
| `depth_trunc_m` | 5.0 | `tiptop.yml` | B |
| `time_dilation_factor` | 0.2 | `tiptop.yml` | B |
| gripper action | 20 steps @ 15 Hz | `tiptop_websocket.py:41` | **F** source of truth |
| extrinsics correction | z −= 0.015 | `tiptop_websocket.py:250` | **F** source of truth |
| `include_workspace` | False | matches `tiptop_websocket_server.py:56` sim default (skips real-robot workspace cuboids for both arms) | B |

## Policy

No `constants.py` refactor: (1) the highest-risk values live upstream and can't be absorbed; (2) `place_propose.py` already names its policy constants at module level and the perception knobs are keyword defaults — both discoverable and overridable; (3) the pipeline has bit-exact regression expectations (G-20) and deliberately-uncommitted validated diffs — cosmetic churn there costs more than it buys. This file is the single registry; update it when a constant is added, retuned, or an upstream mirror (class F) changes.
