#!/usr/bin/env bash
# GWM arm of the scene-6 place eval: 4 tasks x $TRIALS trials on scene 6
# VARIANT 1 (held block welded in the gripper), DEFAULT speed tier (videos on).
# Selection already happened offline (run_place_score.sh,
# winner_place_<tag>.json); per task a fixed-plan policy server on :8770
# serves that winner. Resumes from CSVs; one retry per task.
#
# TRIALS defaults to 1: the arm replays one fixed plan and the scene has no
# randomization, so trial 0 is deterministic (G-16: same plan + same trial
# index -> identical physics outcome); extra trials re-measure nothing.
set -u
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
cd "$(dirname "$0")"
source place_tasks.sh
PY=../droid-sim-evals/.venv/bin/python
PROP=${PROP:-/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_place_v2}
OUT=${OUT:-runs/place_v2}
mkdir -p "$OUT"
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}
TRIALS=${TRIALS:-1}
FAST=${FAST:-}
ARM=${ARM:-gwm}
WSUF=${WSUF:-}
PORT=8770

for tag in "${TAGS[@]}"; do
    csv="$OUT/results_${ARM}_$tag.csv"
    winner="$PROP/winner_place_$tag$WSUF.json"
    if [ ! -f "$winner" ]; then echo "MISSING WINNER $winner — skip $tag"; continue; fi
    for attempt in 1 2; do
        n=$( [ -f "$csv" ] && grep -c "^place_$tag," "$csv" || echo 0 )
        [ "$n" -ge "$TRIALS" ] && break
        echo "======== $ARM place_$tag attempt $attempt (have $n/$TRIALS) $(date +%H:%M:%S) ========"
        fuser -k -s $PORT/tcp 2>/dev/null; sleep 1
        $PY ../gwm_tiptop/policy_server.py --select fixed --plan-file "$winner" \
            --port $PORT --log-jsonl "$OUT/served_${ARM}_$tag.jsonl" \
            > "$OUT/policy_server_${ARM}_$tag.log" 2>&1 &
        PS_PID=$!
        for i in $(seq 1 20); do ss -tln | grep -q ":$PORT " && break; sleep 1; done
        timeout -s KILL "$TASK_TIMEOUT" $PY -u place_eval.py \
            --task-id "place_$tag" --scene 6 --variant 1 \
            --instruction "${INSTR[$tag]}" --success-rule "${RULE[$tag]}" \
            --trials "$TRIALS" --results-csv "$csv" --video-dir "$OUT/videos_${ARM}_$tag" \
            --ws-port $PORT ${FAST}
        rc=$?
        kill $PS_PID 2>/dev/null
        n=$( [ -f "$csv" ] && grep -c "^place_$tag," "$csv" || echo 0 )
        echo "TASKDONE $ARM place_$tag attempt=$attempt rc=$rc rows=$n"
    done
done
fuser -k -s $PORT/tcp 2>/dev/null
echo "$ARM PLACE ARM DONE $(date +%H:%M:%S)"
