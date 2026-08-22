#!/usr/bin/env bash
# Four-system evaluation on scene 6: 10 pick tasks (variant 0) + 4 place tasks
# (variant 1), 5 trials each, DEFAULT speed tier (videos on, NOT --fast).
# Everything lands in ONE directory (default runs/eval_4way) as
# results_<arm>_<tag>.csv; pick task ids are refer6_* and place ids are place_*,
# so the two families share the folder without colliding.
#
# Arms (the three GWM ones differ only in which scoring viewpoint chose the
# plan; the replayed pipeline is identical):
#   gwmfusion  winner_*_fusion.json  two-camera fused scoring
#   gwmcam1    winner_*_cam1.json    external_cam only
#   gwmcam2    winner_*.json         external_cam_2 only (the default)
#   tiptop     no winner file        upstream TiPToP, replanning every trial
#
# Phases 1-3 (the GWM arms) need no GPU planner — a fixed-plan policy server on
# :8770 serves the winner. Phase 4 needs M2T2 (:8123) and tiptop-server (:8765)
# and is not started here: run it separately once those servers are up.
#
#   OUT=runs/eval_4way TRIALS=5 ./run_eval_4way.sh          # phases 1-3
#   OUT=runs/eval_4way TRIALS=5 ARM=tiptop bash run_refer6_tiptop.sh   # phase 4a
#   OUT=runs/eval_4way TRIALS=5 ARM=tiptop bash run_place_tiptop.sh    # phase 4b
set -u
cd "$(dirname "$0")"
OUT=${OUT:-runs/eval_4way}
TRIALS=${TRIALS:-5}
export OUT TRIALS
mkdir -p "$OUT"

for spec in "gwmfusion:_fusion" "gwmcam1:_cam1" "gwmcam2:"; do
    arm=${spec%%:*}
    wsuf=${spec#*:}
    echo "############ ARM $arm (winner suffix '${wsuf}') $(date +%H:%M:%S) ############"
    ARM=$arm WSUF=$wsuf bash run_refer6_gwm.sh
    ARM=$arm WSUF=$wsuf bash run_place_gwm.sh
done
echo "GWM ARMS DONE $(date +%H:%M:%S) — phase 4 (tiptop) must be launched separately"
