# GI-1 Light Reproduction Log (original TiPToP on droid-sim)

*2026-08-09, RTX 3090, websocket mode. Servers: tiptop (ws://8765) + M2T2 (:8123) + IsaacLab client co-resident on one GPU, ~8.4 GB peak. Stock (Gemini + SAM2) pipeline, native pick-and-place instructions.*

## Episodes

| Scene / variant | Instruction | Outcome | Cause / notes |
| --- | --- | --- | --- |
| 1 / 0 | Put the Rubik's cube in the bowl. | ❌ execution failure | plan found (9 steps); grasp slipped, cube left on table — TiPToP's documented top failure mode (25.1 % of trials in their MolmoSpaces run) |
| 1 / 1 | Put the Rubik's cube in the bowl. | ✅ success | cube in bowl |
| 2 / 0 | Put the can in the mug. | ✅ success | can in mug (mug tilted during place) |
| 3 / 0 | Put the banana in the bin. | ✅ success | banana in bin |
| 4 / 0 | Put the cube on the mug and the cans in the bowl. | ❌ planning failure | two-goal task; "No satisfying particles found after optimizing all 10 plan(s)" after the 60 s budget (69.3 s total) — their #2 failure class (optimization timeout, 31 % of no-plan episodes) |
| 4 / 1 | Put the cube on the mug and the cans in the bowl. | ❌ planning failure | retry on another variant: same class ("No satisfying particles after optimizing all 8 plan(s)", 60.7 s) — the two-goal task consistently times out at 256 particles / 60 s budget |
| 5 / 0 | Put 3 blocks in the bowl. | ✅ success | Gemini grounded "3 blocks" → On(yellow/red/green_block, red_bowl); all three placed, episode near the 90 s cap |

**5 scenes covered, 4/7 episodes succeeded; all failures match TiPToP's published failure taxonomy.** Scene 4's two-goal instruction never planned within budget on this hardware (2/2 variants) — irrelevant for our pick-only v1, and a knob (particles / planning budget) exists if a scene-4 baseline is ever needed. Success judged manually from final video frames (no auto-detection yet — that lands with the G-13 fork).

## Latency (single 3090, all services co-resident)

- IsaacLab startup: ~1.5–2 min per invocation (the G-13 batch runner removes this per-episode cost)
- Planning request end-to-end: 11.5–69.3 s observed (scene 5 fastest at 11.5 s; failed scene 4 burned the full 60 s cuTAMP budget); first-ever request 27.6 s with cuTAMP itself at 5.8 s
- Execution: plans finish in ~450–1250 of 1350 max sim steps; sim steps at ~4.5 it/s wall-clock
- Whole episode: ~3–5 min wall-clock

## Environment notes

- **ffmpeg was missing** on the machine → first episode crashed at `mediapy.write_video` and hung in Isaac Sim shutdown (127 threads). Fixed with `apt install ffmpeg`. Symptom to remember: eval process alive at ~100 % CPU, kernel-time heavy, empty `runs/<date>/<time>/` dir.
- Ubuntu 26.04 ran IsaacLab 2.2.0 without issues (officially 22.04/24.04).
- Server-side artifacts per request under `tiptop/tiptop_server_outputs/<timestamp>/` (perception outputs, grasps, cutamp env, plan JSON, metadata); videos under `droid-sim-evals/runs/<date>/<time>/`.

## GI-1 exit status

- ≥1 success or documented failure cause per scene: **scenes 1, 2, 3, 5 succeeded; scene 4 documented (planning failure; variant-1 retry recorded below)** ✅
- Latency notes: ✅ (above)
- User pipeline walkthrough: pending — suggested entry points: a success video (e.g. `runs/2026-08-09/06-24-36/tiptop_scene1_ep0.mp4`), the failed-grasp video (`06-20-44`), and one `tiptop_server_outputs/<ts>/perception/` directory (Gemini bboxes, SAM2 masks, M2T2 grasps visualizations).
