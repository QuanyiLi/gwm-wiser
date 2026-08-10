# droid-sim 50-Trial Reproduction Protocol (paper-faithful)

*Source: TiPToP paper [arXiv:2603.09971v2](https://arxiv.org/html/2603.09971v2) §VII-A + Appendix -C (Table IV rubrics, Table I sim rows). Extracted 2026-08-09; raw HTML cached in the session scratchpad.*

## Protocol

- **5 sim tasks × 10 trials = 50 trials**; object configurations reset **identically** across all trials of a scene (no variant sweep — trials sample pipeline stochasticity: M2T2 sampling, Gemini, cuTAMP particles). We use variant 0 of each scene.
- TiPToP plans **once**, open-loop, wrist camera at capture pose; planning budget 30–60 s (we keep the server default 60 s / 256 particles). Paper sim ran on an L4; we're on a 3090.
- Metrics: binary success rate (SR) per task + Task Progress (TP: 25 % approach / 50 % grasp / 75 % approach target with object / 100 % place; color-cubes scored per cube). v1 of our runner records SR automatically; TP later if needed.

## Task table (paper instructions, verbatim)

| # | Scene | Instruction | Paper TiPToP SR | Paper π0.5-DROID SR |
|---|---|---|---|---|
| 1 | scene 1 var 0 | put the cube in the bowl | 5/10 | 8/10 |
| 2 | scene 2 var 0 | put the can in the mug | 9/10 | 2/10 |
| 3 | scene 3 var 0 | put banana in the bin | 0/10 | 9/10 |
| 4 | scene 4 var 0 | put the meat can on the sugar box | 5/10 | 0/10 |
| 5 | scene 5 var 0 | put 3 cubes into the bowl | 9/10 | 0/10 |

**Aggregate target: 28/50 (56 %)** (computed from rows; the paper reports no sim-only aggregate). Wall-clock reference (successful trials): cube→bowl 17.9 s, can→mug 18.6 s total incl. ~9.5 s perception+planning.

⚠️ The droid-sim-evals README example instructions differ from the paper's Table IV (README scene 4 is a two-goal task that times out; the paper task is single-goal). **Reproduction uses the paper instructions above.**

## Runner

`droid-sim-evals/trial_eval.py` + `run_trials_paper.sh` (our fork additions, G-13 ③): **one process per trial** on tiptop_eval's proven code path (verbatim copy + ~15 appended lines: sim-state success check + CSV append), driven by a bash loop with a SIGKILL watchdog (700 s), one retry per trial, and CSV-based resume. Serves both arms: original TiPToP (websocket :8765) and GWM-TiPToP (same protocol, different server).

*History note (2026-08-10):* a first-generation `batch_eval.py` kept IsaacLab resident across trials/scenes; cross-scene env recreation deterministically hung IsaacLab and the file was removed. A hardened per-SCENE batch (`batch_eval_v2.py` — one boot per task, never switches scenes, shared CSV schema with trial_eval, try/except scoring, heartbeat logs, no output piping) was reintroduced and first exercised on the meat-can task; it saves ~2 min boot overhead per trial (~40 % wall-clock on a 10-trial task) and is the intended runner for the M2 A/B (400+ trials ≈ 14 h of boot overhead saved). Two other hang post-mortems for the record: a success-rule pattern that matched no rigid body (scene 3's bin is named `small_KLT_visual_collision`) raised in the appended scoring block and hung the interpreter at exit behind IsaacLab's threads; and `timeout -s KILL` cannot reach the python when launched via `uv run` (KILL is unforwardable — call `.venv/bin/python` directly). **Cameras/videos stay on in all eval modes** — recordings are a required deliverable; the idea of disabling the external cameras for speed was considered and rejected (2026-08-10). *Revised later the same day (user):* `batch_eval_v2 --fast` adds an opt-in first-frame-only mode — render through reset/settle/planning, then force `render_mode = NO_RENDERING` (direct attribute write; the official setter refuses headless mode changes), restore before the next trial's reset; no videos. Success judging reads sim state and is unaffected. For mass runs where recordings are waived (e.g. the M2 A/B); paper-repro runs keep videos on. Background: camera `update_period=1.0` only throttles sensor-buffer fetches — the RTX renderer still ticks every `render_interval` step while RTX sensors exist, which is why full-rate and 1 Hz walls were similar. Success rules (sim-state):

- *in bowl/bin/mug*: object XY within container footprint, object bottom z between container bottom and rim.
- *on sugar box*: object XY within box footprint, object bottom z ≈ box top ± 2 cm.
- *3 cubes*: rule applied to all three cube bodies.

Judgment note: the paper doesn't state whether sim SR was scored automatically or by a human; our automatic rule is stated here for reproducibility and spot-checked against videos.

## Scoring-validity postmortem (2026-08-10)

Two sim quirks made the first 50-trial round's scene-4/5 numbers garbage; both are fixed by `SuccessTracker` in `batch_eval_v2.py` (mesh-center offset calibrated from USD bbox at settle + per-step root-pose snapshots + judge from the last pre-reset snapshot):

1. **Off-mesh physics pivots**: several assets have `root_pos_w` far from the mesh — scene-5 `_24_bowl` 36.5 cm, scene-4 `_10_potted_meat_can` 27.4 cm, scene-5 red/green blocks 13–15 cm (scene-1/2/3 assets are centered; probe: scratchpad `probe_scene.py`). Raw pivot distances are meaningless there. The recurring meatcan `z_rel=0.016` was exactly the spawn pivot offset (can never lifted).
2. **Truncation auto-reset**: episodes that hit the 90 s cap are reset *inside* `env.step()`, so post-loop judging sees a fresh spawn scene. Only cubes3_bowl episodes (3 pick-places, ~1350 steps) hit this; its first-round 0/4 "identical failures" were resets, and a video trial proved the actual outcome was 3-cubes-in-bowl success.

Validity of round 1: cube_bowl 6/10, can_mug 9/10, banana_bin 0/10 **valid** (centered pivots, episodes end well under the cap); meatcan_sugarbox and cubes3_bowl **invalid → rerun** with the fixed judge (archived in `runs/archive/invalid_pivot_scoring/`).

Variance note (corrects the stochasticity claim above): with this stack the only effective per-trial variance source is **Gemini sampling** — M2T2 is a deterministic forward pass, cuRobo's seed is fixed (`reset(reset_seed=False)`), and cuTAMP particle init is deterministic given identical perception. Small Gemini jitter often collapses to the same discrete grasp/plan choice, so per-task trial outcomes cluster hard (relevant to any A/B power analysis that assumes independent trials).

## Final results (2026-08-10, fixed judge, `runs/batch_paper_fast/`)

| Task | Ours | Paper | Δ |
|---|---|---|---|
| cube_bowl | 6/10 | 5/10 | +1 |
| can_mug | 9/10 | 9/10 | 0 |
| banana_bin | 0/10 | 0/10 | 0 |
| meatcan_sugarbox | 3/10 | 5/10 | −2 |
| cubes3_bowl | 7/10 | 9/10 | −2 |
| **Aggregate** | **25/50 (50 %)** | **28/50 (56 %)** | −3 |

Run mode: `--fast` (~105 s/trial normal tasks, ~250 s cubes3; no videos). One cubes3 trial was re-run after a transient Gemini 503 (infra failures don't count against SR). cubes3 #7 is borderline (third cube snapshot-judged 10.6 cm above the bowl at the 90 s cap, inside the z window; strict reading 6/10). Failure modes seen: grasp whiffs/nudges (meatcan 5, all banana trials far-miss), mid-transfer drops to a consistent spot (~(0.62, 0.19); meatcan 4), third-place timeout at the cap (cubes3 3).
