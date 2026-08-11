#!/usr/bin/env bash
# Score the 4 place instructions against the scene6_place_v2 candidates on a
# running gwm-server (:8901). One score_client call per instruction; winners
# land as proposals/scene6_place_v2/winner_place_<tag>.json, RAT strips under
# runs/place_v2/rat/. Instructions must match place_tasks.sh verbatim.
#
# v2 = perception-only proposer (2026-08-11 audit): candidates cover EVERY
# perceived cluster, not just the two bins, so the scorer must reject
# non-container destinations on its own. v1 artifacts (GT-target proposer)
# stay under proposals/scene6_place + runs/place_v1 for provenance.
#
# Plans here are ~7.5 s (< SCHEDULE[-1]*3.0 = 8.85 s), so sample_rat_times'
# shrink-to-fit branch compresses the window to the whole plan: unlike refer6,
# the last RAT frame is the arm holding the block inside the chosen bin.
set -uo pipefail
PY=/root/code/gwm/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python
PROP=/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_place_v2
H5=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/scenes/captures/scene6_1/external_obs.h5
DBG=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/runs/place_v2/rat
cd /root/code/gwm/gwm-wiser

source /root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/place_tasks.sh
CAM=${CAM:-external_cam_2}   # comma-separate for multi-view fusion
SUF=${SUF:-}                 # artifact suffix, keeps viewpoint variants apart

for tag in "${TAGS[@]}"; do
    echo "=== place_$tag$SUF: ${INSTR[$tag]} (cam $CAM) ==="
    $PY -m gwm_tiptop.score_client --proposals-dir "$PROP" --external-h5 "$H5" --cam "$CAM" \
        --instruction "${INSTR[$tag]}" --tag "place_$tag$SUF" --dump-dir "$DBG$SUF/$tag" || echo "SCORE FAILED: $tag"
done
echo "ALL SCORING DONE"
