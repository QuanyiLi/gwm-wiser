#!/usr/bin/env bash
# tiptop arm of the scene-6 place eval: 4 tasks x $TRIALS trials on scene 6
# VARIANT 1 (held block welded in the gripper). Needs M2T2 (:8123) and
# tiptop-server (:8765) up.
#
# Why this goes through policy_server --select proxy instead of talking to
# :8765 directly (as the pick arm does): the block is welded to the gripper for
# the whole episode, so tiptop's native plan tail (gripper open + GoToInitial)
# carries the block back to the home pose and every trial would score False
# regardless of how well the instruction was grounded -- the number would
# measure the weld, not tiptop. The proxy keeps tiptop's per-trial perception +
# Gemini grounding + cuTAMP planning completely unchanged and only truncates
# the served plan at the last `Place(...)` trajectory step, which is exactly
# where the GWM place candidates end (block inside the chosen bin, still held).
# Both arms are then judged on the same episode shape by the same PlaceTracker.
# The truncation is logged per request in served_tiptop_<tag>.jsonl (full step
# label list + kept count), so the adaptation is auditable per trial.
set -u
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
cd "$(dirname "$0")"
source place_tasks.sh
PY=../droid-sim-evals/.venv/bin/python
OUT=${OUT:-runs/place_v2}
mkdir -p "$OUT"
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}
TRIALS=${TRIALS:-5}
FAST=${FAST:-}
ARM=${ARM:-tiptop}
PORT=8770

for tag in "${TAGS[@]}"; do
    csv="$OUT/results_${ARM}_$tag.csv"
    for attempt in 1 2; do
        n=$( [ -f "$csv" ] && grep -c "^place_$tag," "$csv" || echo 0 )
        [ "$n" -ge "$TRIALS" ] && break
        echo "======== TIPTOP place_$tag attempt $attempt (have $n/$TRIALS) $(date +%H:%M:%S) ========"
        fuser -k -s $PORT/tcp 2>/dev/null; sleep 1
        $PY ../gwm_tiptop/policy_server.py --select proxy --upstream-port 8765 \
            --truncate-after Place --port $PORT --log-jsonl "$OUT/served_${ARM}_$tag.jsonl" \
            > "$OUT/proxy_server_${ARM}_$tag.log" 2>&1 &
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
        echo "TASKDONE tiptop place_$tag attempt=$attempt rc=$rc rows=$n"
    done
done
fuser -k -s $PORT/tcp 2>/dev/null
echo "TIPTOP PLACE ARM DONE $(date +%H:%M:%S)"
