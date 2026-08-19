#!/usr/bin/env bash
# Services for the GWM arm.
#
#   ./droid/gwm_hardware/gwm_arm/services.sh start [dummy|gwm]
#   ./droid/gwm_hardware/gwm_arm/services.sh status
#   ./droid/gwm_hardware/gwm_arm/services.sh stop
#
# The GWM arm needs everything the baseline arm needs (M2T2 for grasps,
# FoundationStereo for depth, the camera pre-flight, the sm_120 PTX warm-up)
# plus one more: gwm-server, which owns the pinned transformers==4.57.6
# environment and cannot share a process with the tiptop stack (D9). So this
# script defers the shared half to the baseline arm's script rather than
# duplicating it, and adds only what is ours.
#
# Two backends:
#   dummy  renders the real RAT frames and scores with a hash of the
#          trajectory. Selection is meaningless BY DESIGN; use it to prove the
#          HTTP path, the renderer seam and the artefacts, without paying 20 GB
#          of VRAM or waiting for the Qwen weights.
#   gwm    the real thing: Qwen3-VL-Embedding-8B + the trained GWM checkpoint.
#
# Killing by port, never by `kill $!` -- a 20 GB orphaned scorer starved a whole
# evening of eval once (G-25).
set -uo pipefail

ROOT=/home/quanyi/gwm-wiser
LOGS=$ROOT/droid/gwm_hardware/.service-logs
CKPT=${GWM_CKPT:-/home/quanyi/0810_gwm/checkpoint.pt}
PORT=${GWM_PORT:-8901}
URDF=$ROOT/droid/gwm_hardware/assets/panda_robotiq_2f_140_render.urdf
export PATH="$HOME/.pixi/bin:$PATH"

up() { curl -sf --max-time 3 "http://localhost:$1/health" >/dev/null 2>&1; }

status() {
    "$ROOT/droid/gwm_hardware/tiptop_arm/services.sh" status
    if up "$PORT"; then
        echo "  gwm-server ($PORT)  up  -- $(curl -s "http://localhost:$PORT/health")"
    else
        echo "  gwm-server ($PORT)  DOWN"
    fi
}

start() {
    local backend=${1:-gwm}
    mkdir -p "$LOGS"
    # Shared perception stack + camera pre-flight + PTX warm-up.
    "$ROOT/droid/gwm_hardware/tiptop_arm/services.sh" start || return 1

    # The render model is derived from the planning URDF; rebuild if stale.
    "$ROOT/.venv/bin/python" -c "
import sys; sys.path[:0] = ['$ROOT', '$ROOT/droid']
from gwm_hardware.gwm_arm.render_model import ensure_render_urdf
print(ensure_render_urdf())" || return 1

    if up "$PORT"; then
        echo "gwm-server already up on $PORT"
    else
        if [ "$backend" = gwm ] && [ ! -f "$CKPT" ]; then
            echo "checkpoint $CKPT not found (set GWM_CKPT)"; return 1
        fi
        echo "starting gwm-server (backend=$backend)..."
        ( cd "$ROOT" && PYTHONPATH="$ROOT:$ROOT/droid" nohup ./.venv/bin/python \
            -m droid.server.gwm_server --backend "$backend" --arm panda \
            --urdf "$URDF" ${CKPT:+--ckpt "$CKPT"} --port "$PORT" \
            --head-dtype "${GWM_HEAD_DTYPE:-bf16}" \
            >"$LOGS/gwm_server.log" 2>&1 & )
        echo "waiting for gwm-server (the Qwen weights take a few minutes cold)..."
        for _ in $(seq 1 240); do up "$PORT" && break; sleep 5; done
        up "$PORT" || { echo "gwm-server did not come up; see $LOGS/gwm_server.log"; return 1; }
    fi
    echo; status
}

stop() {
    fuser -k "$PORT/tcp" 2>/dev/null && echo "gwm-server ($PORT) stopped"
    "$ROOT/droid/gwm_hardware/tiptop_arm/services.sh" stop
}

case "${1:-status}" in
    start)  start "${2:-gwm}" ;;
    stop)   stop ;;
    status) status ;;
    *)      echo "usage: $0 {start [dummy|gwm]|stop|status}"; exit 1 ;;
esac
