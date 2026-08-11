#!/usr/bin/env bash
# tiptop arm of the scene-6 referral eval: 10 tasks x 10 trials, DEFAULT speed
# tier (NOT --fast; 1 Hz cameras + per-trial videos). Needs M2T2 (:8123) and
# tiptop-server (:8765) up. Resumes from CSVs; one retry per task.
set -u
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
cd "$(dirname "$0")"
source refer6_tasks.sh
PY=../droid-sim-evals/.venv/bin/python
OUT=${OUT:-runs/refer6}
mkdir -p "$OUT"
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}
TRIALS=${TRIALS:-10}
FAST=${FAST:-}          # set to --fast for mass runs (rendering stays on until the plan arrives)

for tag in "${TAGS[@]}"; do
    csv="$OUT/results_tiptop_$tag.csv"
    for attempt in 1 2; do
        n=$( [ -f "$csv" ] && grep -c "^refer6_$tag," "$csv" || echo 0 )
        [ "$n" -ge "$TRIALS" ] && break
        echo "======== TIPTOP $tag attempt $attempt (have $n/$TRIALS) $(date +%H:%M:%S) ========"
        timeout -s KILL "$TASK_TIMEOUT" $PY -u grasp_eval.py \
            --task-id "refer6_$tag" --scene 6 --variant 0 \
            --instruction "${INSTR[$tag]}" --success-rule "${RULE[$tag]}" \
            --trials "$TRIALS" --results-csv "$csv" --video-dir "$OUT/videos_tiptop_$tag" ${FAST}
        rc=$?
        n=$( [ -f "$csv" ] && grep -c "^refer6_$tag," "$csv" || echo 0 )
        echo "TASKDONE tiptop $tag attempt=$attempt rc=$rc rows=$n"
    done
done
echo "TIPTOP ARM DONE $(date +%H:%M:%S)"
