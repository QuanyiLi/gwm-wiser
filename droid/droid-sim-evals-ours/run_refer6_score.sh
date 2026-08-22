#!/usr/bin/env bash
# Offline GWM selection for the 10 scene-6 referring-expression tasks, against
# the scene6_rev2 candidate pool (16 whole-scene candidates, perception-only
# proposer) on a running gwm-server (:8901). Two steps per tag:
#   1. score_client  -- object chosen by the MEAN of its candidates' GWM scores
#                       (--object-score mean), winner = that object's best
#                       candidate -> winner_refer6_<tag>.json. The mean is
#                       preferred over the global per-candidate argmax
#                       (--object-score max) because se3_fps_indices samples
#                       each object's small quota at the EXTREMES of its grasp
#                       family, so a single best candidate is a noisy proxy
#                       for the object.
#   2. grasp_gate    -- closing-line veto within the WINNING OBJECT; re-picks
#                       the best PASSING plan of that object and keeps the
#                       ungated winner when the object has none.
# Wordings come from refer6_tasks.sh so the scorer and the runners cannot
# drift apart. RAT strips land under runs/refer6_rev2/rat/.
set -uo pipefail
PY=/root/code/gwm/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python
PROP=/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_rev2
H5=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/scenes/captures/scene6_0/external_obs.h5
WRIST_H5=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/scenes/captures/scene6_0/wrist_obs.h5
GATE=${GATE:-1}
# CAM/SUF exist for viewpoint ablations: the capture h5 stores BOTH external
# cameras, and gwm-server takes K/c2w per request, so re-scoring from the other
# side costs one server run and no re-capture. SUF keeps the artifacts of
# different viewpoints apart (scores_refer6_<tag><SUF>.json).
CAM=${CAM:-external_cam_2}
SUF=${SUF:-}
DBG=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/runs/refer6_rev2/rat$SUF

source /root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/refer6_tasks.sh
cd /root/code/gwm/gwm-wiser

for tag in "${TAGS[@]}"; do
    echo "=== refer6_$tag$SUF: ${INSTR[$tag]} (cam $CAM) ==="
    $PY -m gwm_tiptop.score_client --proposals-dir "$PROP" --external-h5 "$H5" --cam "$CAM" \
        --instruction "${INSTR[$tag]}" --tag "refer6_$tag$SUF" --dump-dir "$DBG/$tag" \
        || { echo "SCORE FAILED: $tag"; continue; }
    [ "$GATE" = "1" ] && $PY -m gwm_tiptop.grasp_gate --proposals-dir "$PROP" \
        --h5-path "$WRIST_H5" --apply "refer6_$tag$SUF" || echo "GATE FAILED: $tag"
done
echo "ALL SCORING DONE"
