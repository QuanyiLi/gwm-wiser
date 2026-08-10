# our-droid-sim-evals

Custom DROID-sim evaluation tasks for TiPToP, layered on [`../droid-sim-evals`](../droid-sim-evals)
(same IsaacLab `.venv`, same runner, same CSV schema — `grasp_eval.py` imports `batch_eval_v2`
and swaps in a lift-based success judge).

## Tasks (scene 1)

| task id | instruction | success rule |
|---|---|---|
| grasp_cube | pick up the cube | cube center ≥ 0.15 m above its settled height at trial end |
| grasp_bowl | pick up the bowl | bowl center ≥ 0.15 m above its settled height at trial end |

TiPToP handles these natively: Gemini grounds "pick up X" to a `holding(x)` atom and cuTAMP
plans against a `Holding` goal (no place), ending with retract + go-home while still gripping —
a successful grasp carries the object well above the lift threshold, a dropped one falls back
during the 30-step post-plan hold.

## Run

Servers must be up first (M2T2 on :8123, tiptop-server on :8765 — see [`../README.md`](../README.md)).

```bash
PHASE=smoke ./run_grasp_tasks.sh   # 1 trial per task, with video — eyeball first
PHASE=batch ./run_grasp_tasks.sh   # fill to $TRIALS (default 5) in --fast, resumes from CSV
./run_grasp_tasks.sh               # both phases back to back
```

Results land in `runs/grasp_v1/results_<task>.csv` (schema `task,trial,success,plan_failed,detail`),
smoke videos in `runs/grasp_v1/videos_<task>/`.
