#!/usr/bin/env bash
# tiptop arm of the scene-6 place eval: 4 tasks x $TRIALS trials on scene 6
# VARIANT 1 (held block welded in the gripper). Needs M2T2 (:8123) and
# tiptop-server (:8765) up.
#
# tiptop plans natively here: it picks the block, carries it to the bin, opens
# the gripper and goes home. `weld_held_block` releases the weld the first time
# the gripper reopens after closing, so the released block falls into the bin
# instead of being carried home.
#
# TRUNCATE=Place selects an alternative protocol: it routes through
# `policy_server --select proxy`, which forwards to :8765 unchanged and only
# drops the plan tail after the last `Place(...)` trajectory step, ending the
# episode where the GWM candidates end (block inside the bin, still held).
# Each proxied request is logged to served_<arm>_<tag>.jsonl with the full step
# list and the kept count.
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
TRUNCATE=${TRUNCATE:-}    # empty = native tiptop; 'Place' = proxy + truncate
PORT=8770

for tag in "${TAGS[@]}"; do
    csv="$OUT/results_${ARM}_$tag.csv"
    for attempt in 1 2; do
        n=$( [ -f "$csv" ] && grep -c "^place_$tag," "$csv" || echo 0 )
        [ "$n" -ge "$TRIALS" ] && break
        echo "======== TIPTOP place_$tag attempt $attempt (have $n/$TRIALS) $(date +%H:%M:%S) ========"
        WS_PORT=8765
        PS_PID=
        if [ -n "$TRUNCATE" ]; then
            fuser -k -s $PORT/tcp 2>/dev/null; sleep 1
            $PY ../gwm_tiptop/policy_server.py --select proxy --upstream-port 8765 \
                --truncate-after "$TRUNCATE" --port $PORT --log-jsonl "$OUT/served_${ARM}_$tag.jsonl" \
                > "$OUT/proxy_server_${ARM}_$tag.log" 2>&1 &
            PS_PID=$!
            for i in $(seq 1 20); do ss -tln | grep -q ":$PORT " && break; sleep 1; done
            WS_PORT=$PORT
        fi
        timeout -s KILL "$TASK_TIMEOUT" $PY -u place_eval.py \
            --task-id "place_$tag" --scene 6 --variant 1 \
            --instruction "${INSTR[$tag]}" --success-rule "${RULE[$tag]}" \
            --trials "$TRIALS" --results-csv "$csv" --video-dir "$OUT/videos_${ARM}_$tag" \
            --ws-port $WS_PORT ${FAST}
        rc=$?
        [ -n "$PS_PID" ] && kill $PS_PID 2>/dev/null
        n=$( [ -f "$csv" ] && grep -c "^place_$tag," "$csv" || echo 0 )
        echo "TASKDONE tiptop place_$tag attempt=$attempt rc=$rc rows=$n"
    done
done
fuser -k -s $PORT/tcp 2>/dev/null
echo "TIPTOP PLACE ARM DONE $(date +%H:%M:%S)"
