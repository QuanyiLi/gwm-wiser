#!/bin/bash
# Grasp-and-hold eval on scene 1: grasp_cube, grasp_bowl.
# PHASE=smoke  -> 1 trial per task WITH video (eyeball behavior first)
# PHASE=batch  -> fill to $TRIALS in --fast (no videos); resumes from CSV,
#                 so smoke trials never re-run. SIGKILL watchdog + one retry.
# default      -> smoke then batch.
set -u
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$(dirname "$0")"
PY=../droid-sim-evals/.venv/bin/python
OUT=runs/grasp_v1
mkdir -p "$OUT"
TASK_TIMEOUT=${TASK_TIMEOUT:-2400}
TRIALS=${TRIALS:-5}
PHASE=${PHASE:-all}

declare -A INSTR RULE
INSTR[grasp_cube]="pick up the cube"
RULE[grasp_cube]='{"objects":["cube"],"lift":0.15}'
INSTR[grasp_bowl]="pick up the bowl"
RULE[grasp_bowl]='{"objects":["bowl"],"lift":0.15}'

run_task() {  # run_task <task> <trials> <extra flags...>
    local task=$1 trials=$2; shift 2
    timeout -s KILL "$TASK_TIMEOUT" $PY -u grasp_eval.py \
        --task-id "$task" --scene 1 --variant 0 \
        --instruction "${INSTR[$task]}" --success-rule "${RULE[$task]}" \
        --trials "$trials" --results-csv "$OUT/results_$task.csv" \
        --video-dir "$OUT/videos_$task" "$@"
}

if [ "$PHASE" = smoke ] || [ "$PHASE" = all ]; then
    for task in grasp_cube grasp_bowl; do
        echo "======== SMOKE $task (1 trial, video) $(date +%H:%M:%S) ========"
        run_task "$task" 1
    done
fi

if [ "$PHASE" = batch ] || [ "$PHASE" = all ]; then
    for task in grasp_cube grasp_bowl; do
        for attempt in 1 2; do
            echo "======== BATCH $task (attempt $attempt) $(date +%H:%M:%S) ========"
            run_task "$task" "$TRIALS" --fast
            rc=$?
            n=$( [ -f "$OUT/results_$task.csv" ] && grep -c "^$task," "$OUT/results_$task.csv" || echo 0 )
            echo "task $task attempt $attempt rc=$rc rows=$n"
            [ "$n" -ge "$TRIALS" ] && break
        done
    done
fi
echo "PHASE=$PHASE DONE $(date +%H:%M:%S)"
