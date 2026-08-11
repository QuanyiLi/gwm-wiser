#!/usr/bin/env bash
# GWM arm of the scene-6 referral eval: 10 tasks x 10 trials, DEFAULT speed
# tier (NOT --fast; videos on). Selection already happened offline
# (run_refer6_score.sh, winner_refer6_<tag>.json); per task a fixed-plan policy server on
# :8770 serves that winner, so no GPU planner is involved. Resumes from CSVs.
set -u
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
cd "$(dirname "$0")"
source refer6_tasks.sh
PY=../droid-sim-evals/.venv/bin/python
PROP=${PROP:-/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_rev2}
OUT=${OUT:-runs/refer6}
mkdir -p "$OUT"
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}
TRIALS=${TRIALS:-10}
FAST=${FAST:-}          # set to --fast for mass runs (no videos, render gated at plan time)
ARM=${ARM:-gwm}         # names the CSVs/logs, so several selection variants share one OUT dir
WSUF=${WSUF:-}          # winner variant to serve: '' = canonical, _cam1, _fusion, ...
PORT=8770

for tag in "${TAGS[@]}"; do
    csv="$OUT/results_${ARM}_$tag.csv"
    winner="$PROP/winner_refer6_$tag$WSUF.json"
    if [ ! -f "$winner" ]; then echo "MISSING WINNER $winner — skip $tag"; continue; fi
    for attempt in 1 2; do
        n=$( [ -f "$csv" ] && grep -c "^refer6_$tag," "$csv" || echo 0 )
        [ "$n" -ge "$TRIALS" ] && break
        echo "======== $ARM $tag attempt $attempt (have $n/$TRIALS) $(date +%H:%M:%S) ========"
        fuser -k -s $PORT/tcp 2>/dev/null; sleep 1
        $PY ../gwm_tiptop/policy_server.py --select fixed --plan-file "$winner" \
            --port $PORT --log-jsonl "$OUT/served_${ARM}_$tag.jsonl" \
            > "$OUT/policy_server_${ARM}_$tag.log" 2>&1 &
        PS_PID=$!
        for i in $(seq 1 20); do ss -tln | grep -q ":$PORT " && break; sleep 1; done
        timeout -s KILL "$TASK_TIMEOUT" $PY -u grasp_eval.py \
            --task-id "refer6_$tag" --scene 6 --variant 0 \
            --instruction "${INSTR[$tag]}" --success-rule "${RULE[$tag]}" \
            --trials "$TRIALS" --results-csv "$csv" --video-dir "$OUT/videos_${ARM}_$tag" \
            --ws-port $PORT ${FAST}
        rc=$?
        kill $PS_PID 2>/dev/null
        n=$( [ -f "$csv" ] && grep -c "^refer6_$tag," "$csv" || echo 0 )
        echo "TASKDONE $ARM $tag attempt=$attempt rc=$rc rows=$n"
    done
done
fuser -k -s $PORT/tcp 2>/dev/null
echo "$ARM ARM DONE $(date +%H:%M:%S)"
