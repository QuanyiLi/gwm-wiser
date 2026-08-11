#!/usr/bin/env bash
# Score the 10 scene-6 referring-expression instructions against the 24
# scene-6 candidates on a running gwm-server (:8901). One score_client call
# per instruction (default rat_scale 3.0); winners land as
# proposals/scene6/winner_refer6_<tag>.json, RAT strips under runs/refer6/rat/.
set -uo pipefail
PY=/root/code/gwm/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python
PROP=/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6
H5=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/scenes/captures/scene6_0/external_obs.h5
DBG=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/runs/refer6/rat
cd /root/code/gwm/gwm-wiser

declare -A INSTR
INSTR[fruit]="pick up the fruit"
INSTR[yellow]="pick up the yellow object"
INSTR[eat]="pick up the thing you could eat"
INSTR[negation]="pick up the object that is neither a toy nor a container"
INSTR[puzzle]="pick up the puzzle toy"
INSTR[colorful]="pick up the most colorful object"
INSTR[nearbowl]="pick up the object closest to the bowl"
INSTR[eatfrom]="pick up the object you would eat a meal from"
INSTR[between]="pick up the object between the cube and the banana"
INSTR[container]="pick up the container"

for tag in fruit yellow eat negation puzzle colorful nearbowl eatfrom between container; do
    echo "=== refer6_$tag: ${INSTR[$tag]} ==="
    $PY -m gwm_tiptop.score_client --proposals-dir "$PROP" --external-h5 "$H5" \
        --instruction "${INSTR[$tag]}" --tag "refer6_$tag" --dump-dir "$DBG/$tag" || echo "SCORE FAILED: $tag"
done
echo "ALL SCORING DONE"
