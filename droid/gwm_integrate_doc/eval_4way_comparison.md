# Scene-6 four-system evaluation — 10 pick + 4 place, 5 trials each

Results dir `runs/eval_4way`, default speed tier (videos on, not `--fast`).

Arms: three GWM arms replaying the SAME pipeline and differing only in the scoring viewpoint that chose the plan, plus upstream TiPToP replanning every trial. Where two GWM arms select the same plan their trials are expected to be byte-identical (fixed-plan replay of a deterministic scene).


## Pick (scene 6 variant 0) — success = target lifted ≥ 0.15 m

| task | target | GWM fused (cam1+cam2) | GWM cam1 | GWM cam2 | tiptop (baseline) |
|---|---|---|---|---|---|
| fruit | banana | 5/5 | 5/5 | 5/5 | 4/5 |
| yellow | banana | 5/5 | 0/5 | 5/5 | 5/5 |
| eat | banana | 5/5 | 5/5 | 5/5 | 5/5 |
| negation | banana | 5/5 | 5/5 | 5/5 | 5/5 |
| puzzle | cube | 5/5 | 5/5 | 5/5 | 4/5 |
| colorful | cube | 5/5 | 5/5 | 5/5 | 4/5 |
| nearbowl | cube | 5/5 | 5/5 | 5/5 | 4/5 |
| eatfrom | bowl | 5/5 | 5/5 | 5/5 | 5/5 |
| between | bowl | 5/5 | 5/5 | 5/5 | 3/5 |
| container | bowl | 5/5 | 5/5 | 5/5 | 5/5 |

## Place (scene 6 variant 1) — success = block inside the named bin

| task | target | GWM fused (cam1+cam2) | GWM cam1 | GWM cam2 | tiptop (baseline) |
|---|---|---|---|---|---|
| red | red_bin | 5/5 <br><sub>red_bin×5</sub> | 5/5 <br><sub>red_bin×5</sub> | 5/5 <br><sub>red_bin×5</sub> | 4/5 <br><sub>red_bin×4</sub> |
| green | green_bin | 5/5 <br><sub>green_bin×5</sub> | 5/5 <br><sub>green_bin×5</sub> | 5/5 <br><sub>green_bin×5</sub> | 4/5 <br><sub>green_bin×4</sub> |
| tomato | red_bin | 5/5 <br><sub>red_bin×5</sub> | 5/5 <br><sub>red_bin×5</sub> | 5/5 <br><sub>red_bin×5</sub> | 5/5 <br><sub>red_bin×5</sub> |
| grass | green_bin | 5/5 <br><sub>green_bin×5</sub> | 5/5 <br><sub>green_bin×5</sub> | 5/5 <br><sub>green_bin×5</sub> | 5/5 <br><sub>green_bin×5</sub> |

## Totals

| arm | success | trials recorded | of 70 planned |
|---|---|---|---|
| GWM fused (cam1+cam2) | 70/70 (100%) | 70 | 100% |
| GWM cam1 | 65/70 (93%) | 70 | 100% |
| GWM cam2 | 70/70 (100%) | 70 | 100% |
| tiptop (baseline) | 62/70 (89%) | 70 | 100% |

## Offline selection provenance (GWM arms)

| task | target | GWM fused (cam1+cam2) | GWM cam1 | GWM cam2 |
|---|---|---|---|---|
| fruit | banana | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> |
| yellow | banana | ✓ banana<br><sub>plan_13_object_4</sub> | red_bin<br><sub>plan_07_object_2</sub> | ✓ banana<br><sub>plan_13_object_4</sub> |
| eat | banana | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> |
| negation | banana | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> | ✓ banana<br><sub>plan_13_object_4</sub> |
| puzzle | cube | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> |
| colorful | cube | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> |
| nearbowl | cube | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> | ✓ cube<br><sub>plan_10_object_3</sub> |
| eatfrom | bowl | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> |
| between | bowl | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> |
| container | bowl | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> | ✓ bowl<br><sub>plan_04_object_1</sub> |
| place_red | red_bin | ✓ red_bin<br><sub>plan_04_object_1</sub> | ✓ red_bin<br><sub>plan_03_object_1</sub> | ✓ red_bin<br><sub>plan_04_object_1</sub> |
| place_green | green_bin | ✓ green_bin<br><sub>plan_01_object_0</sub> | ✓ green_bin<br><sub>plan_01_object_0</sub> | ✓ green_bin<br><sub>plan_01_object_0</sub> |
| place_tomato | red_bin | ✓ red_bin<br><sub>plan_04_object_1</sub> | ✓ red_bin<br><sub>plan_03_object_1</sub> | ✓ red_bin<br><sub>plan_04_object_1</sub> |
| place_grass | green_bin | ✓ green_bin<br><sub>plan_01_object_0</sub> | ✓ green_bin<br><sub>plan_01_object_0</sub> | ✓ green_bin<br><sub>plan_01_object_0</sub> |

## Plan failures / scoring errors

- tiptop/refer6_fruit trial 0: {"plan_failed": true, "error": "Gemini returned a non-JSON response; check for a discrepancy in your image: {\n    \"bboxes\": [\n        {\"box_2d\": [94, 54, 417, 300], \"label\": \"green_box\"},\n 

## Protocol

Scene 6 (rev2), 5 trials per task per arm, default speed tier (1 Hz cameras +
per-trial videos, **not** `--fast`). One judge, byte-identical for all four
arms: pick = target lifted >= 0.15 m at episode end; place = block's mesh
centre within 0.05 m of the named bin's centre and inside z_rel [-0.03, +0.03].
The place band was **not** tuned for this run (a brief widening to -0.05 was
reverted once a released block measured -0.016 — see G-31).

The three GWM arms share one pipeline and differ ONLY in which scoring
viewpoint chose the plan. They replay a fixed plan, so no planner runs at trial
time. TiPToP replans from scratch every trial (perception + Gemini grounding +
M2T2 + cuTAMP). On the place family the welded block is released the first time
the gripper reopens after closing, so TiPToP runs its native plan (pick, carry,
open, go home) while the GWM candidates — which never reopen the gripper — are
unaffected.

## Failure taxonomy (13 failures / 280 trials)

| arm | n | mode | evidence |
|---|---|---|---|
| GWM cam1 | 5 | **grounding** — reached for the red bin | `yellow` x5, banana `z_rel` 0.000 at its spawn pose every trial |
| tiptop | 3 | grasp slip | cube left at/near its spawn pose (`puzzle` t3, `nearbowl` t1) or shoved 8 cm (`colorful` t4) |
| tiptop | 2 | grounding | `between` t0/t2, bowl untouched at its spawn pose |
| tiptop | 2 | release timing | `red` t3, `green` t0: block still 0.254 m up, carried by the arm |
| tiptop | 1 | LLM output | `fruit` t0: Gemini returned non-JSON (truncated bbox list) |

The two systems fail differently, and that matters more than the totals.
GWM cam1's five failures are **one** systematic defect: a viewpoint in which
the banana is small, distant and gripper-shadowed, so the object choice is
wrong — identically, every trial, forever, until the viewpoint or the fusion
changes. TiPToP's eight failures are spread over six different tasks and never
repeat within a task's five trials: they are resampling variance (grasp pose,
Gemini output, plan timing). Predictable-but-frozen versus unpredictable-but-
self-recovering.

## Determinism

Every task where two arms selected the same plan reproduced trial-for-trial
identical outcomes (210 GWM trials, zero exceptions) — e.g. `fruit`, `yellow`
and `eat` all serve `plan_13` and share `z_rel = [0.230, 0.224, 0.229, 0.229,
0.223]`. This re-confirms G-16 (fixed plan + trial index -> identical physics)
and means the GWM arms only carry independent information on tasks where their
selections differ.

## Precision, where success is not the whole story

On the place family TiPToP lands the block far closer to the bin centre
(5-12 mm) than any GWM arm (14-35 mm), and on `place_red`/`place_tomato` the
cam1 selection (14 mm) beats fusion and cam2 (31 mm). Success rate hides this:
all of them clear the 50 mm tolerance.

## Caveats

- One scene, one layout, 14 instructions, 5 trials. Margins and rates are not
  transferable to a new layout; the scoring viewpoint in particular is a
  first-order factor (G-29) and was chosen post-hoc for the cam2 arm.
- The GWM arms' selection is offline and one-shot; their per-trial cost is a
  replay. TiPToP pays full perception + planning every trial. The comparison is
  fair on task outcome, not on compute.
- `yellow` under cam1 is the only pick task separating the GWM arms, so
  "fusion beats cam1 by 5 trials" rests on a single instruction.

