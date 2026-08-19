#!/usr/bin/env bash
# The GWM arm's front door: bring the whole stack up, then take instructions.
#
#   ./droid/gwm_hardware/gwm_arm/run.sh              # dry run -- proposes and scores, never moves
#   ./droid/gwm_hardware/gwm_arm/run.sh --execute    # arms the robot; every motion still confirmed
#
# Starts M2T2, FoundationStereo, the camera pre-flight, the sm_120 warm-up and
# gwm-server, then hands you a prompt. All four fit on this card at once
# (measured 23.1 GB of 32.6), so nothing is torn down between turns.
set -uo pipefail
ROOT=/home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"

"$ROOT/droid/gwm_hardware/gwm_arm/services.sh" start gwm || exit 1
cd "$ROOT" && exec pixi run --manifest-path droid/tiptop/pixi.toml \
    python -m gwm_hardware.gwm_arm.session "$@"
