#!/usr/bin/env bash
# Score + analyse both scene-6 families under the rollout configurations.
#   w32_s4       32-frame context, 3.75 fps actions (the model's cadence)   <- headline
#   w8_s4        8-frame context (the training clip length)
#   w32_s8       2x temporally subsampled actions (0.53 s steps, half the AR steps)
#   w32_s16      4x subsampled actions (1.07 s steps; beyond the 0.075 cap, ~20 AR steps)
#   w32_s4_cam1  the other external camera (GWM's cam1 ablation viewpoint)
#   w32_s4_tcp   state point at the gripper TCP (flange + 0.1625 m along the tool axis) instead of the flange
# Every config also carries the short-horizon goal banks (1.5 / 3 / 6 s) and, for pick, the lift bank.
# Preprocessing: CROP=full_aa (faithful: whole frame squashed to 256x256) by default; the
# w32_s4_crop135 directories hold the earlier 1.35:1 centre-crop run as an ablation.
# Usage: bash run_scoring.sh [pick|place|all]
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
FAM=${1:-all}
CROP=${CROP:-full_aa}

run_family() {
    local fam=$1 replay=$2 plans=$3
    for cfg in "w32_s4 32 4 external_cam_2 0" "w8_s4 8 4 external_cam_2 0" "w32_s8 32 8 external_cam_2 0" \
               "w32_s16 32 16 external_cam_2 0" "w32_s4_cam1 32 4 external_cam 0" "w32_s4_tcp 32 4 external_cam_2 0.1625"; do
        set -- $cfg
        local tag=$1 win=$2 stride=$3 cam=$4 tcp=$5
        local out=runs/vjepa_$fam/$tag
        if [ ! -f "$out/energies.npz" ]; then
            echo "=== score $fam $tag ==="
            $PY score_vjepa.py --family "$fam" --replay-dir "$replay" --plans-dir "$plans" \
                --out-dir "$out" --window "$win" --stride "$stride" --cam "$cam" --tcp-offset "$tcp" --crop-mode "$CROP" \
                2>&1 | grep --line-buffered -v -i "futurewarning\|warnings.warn\|UserWarning\|tensor_numpy" | tee "logs/score_${fam}_${tag}.log"
        fi
        echo "=== analyse $fam $tag ==="
        $PY analyze_selection.py --family "$fam" --energy-dir "$out" --plans-dir "$plans" --tag "$tag" \
            > "logs/analyze_${fam}_${tag}.log" 2>&1 || echo "ANALYSE FAILED: $fam $tag"
        grep -E "^\| (pred|oracle) \| (final:final|h3:at_h|h6:at_h) \| (argmin|two_stage)" "$out/summary.md"
    done
    # the preprocessing ablation (older energies, no horizon banks): re-analyse only
    if [ -f "runs/vjepa_$fam/w32_s4_crop135/energies.npz" ]; then
        $PY analyze_selection.py --family "$fam" --energy-dir "runs/vjepa_$fam/w32_s4_crop135" --plans-dir "$plans" --tag w32_s4_crop135 \
            > "logs/analyze_${fam}_w32_s4_crop135.log" 2>&1 || echo "ANALYSE FAILED: $fam crop135"
    fi
}

[ "$FAM" = pick ] || [ "$FAM" = all ] && run_family pick runs/replay_pick ../gwm_integrate_doc/proposals/scene6_rev2
[ "$FAM" = place ] || [ "$FAM" = all ] && run_family place runs/replay_place ../gwm_integrate_doc/proposals/scene6_place_v2
echo "SCORING DONE"
