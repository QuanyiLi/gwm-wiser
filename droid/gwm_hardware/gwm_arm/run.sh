#!/usr/bin/env bash
# The GWM arm's front door: bring the whole stack up, then take instructions.
#
#   ./droid/gwm_hardware/gwm_arm/run.sh              # dry run -- proposes and scores, never moves
#   ./droid/gwm_hardware/gwm_arm/run.sh --execute    # arms the robot; every motion still confirmed
#   ./droid/gwm_hardware/gwm_arm/run.sh --execute --record --debias-prior
#
# Every flag is forwarded to gwm_arm.session; --help lists them. The two worth
# knowing: --record keeps a video of what the plan actually did, and
# --debias-prior ranks on score MINUS each candidate's instruction-independent
# prior. The prior is recorded and printed either way.
#
# Starts M2T2, FoundationStereo, the camera pre-flight, the sm_120 warm-up and
# gwm-server, then hands you a prompt. All four fit on this card at once, so
# nothing is torn down between turns.
set -uo pipefail
ROOT=/home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"

"$ROOT/droid/gwm_hardware/gwm_arm/services.sh" start gwm || exit 1
cd "$ROOT" && exec pixi run --manifest-path droid/tiptop/pixi.toml \
    python -m gwm_hardware.gwm_arm.session "$@"
