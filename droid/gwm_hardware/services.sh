#!/usr/bin/env bash
# Bring the perception services up, warm, and verified -- then hand you tiptop.
#
#   ./droid/gwm_hardware/services.sh start     # servers + preflight + warm-up
#   ./droid/gwm_hardware/services.sh status
#   ./droid/gwm_hardware/services.sh stop
#   ./droid/gwm_hardware/services.sh run       # start (if needed), then tiptop-run
#
# Order matters and the reasons are rig-specific:
#   * rs_preflight before anything: the RealSenses come up with IR
#     auto-exposure off, which saturates the stereo pair and guts the depth.
#   * warm_servers after the servers are healthy: M2T2 and FoundationStereo pin
#     torch 2.4.1 / CUDA 12.0, which carries no sm_120 cubins, so Blackwell JITs
#     every kernel on the first call -- 33 s against tiptop's hard-coded 10 s
#     client timeout. One throwaway request per server per restart fixes it.
set -uo pipefail

ROOT=/home/quanyi/gwm-wiser
LOGS=$ROOT/droid/gwm_hardware/.service-logs
export PATH="$HOME/.pixi/bin:$PATH"
P="pixi run --manifest-path $ROOT/droid/tiptop/pixi.toml"

up() { curl -sf --max-time 3 "http://localhost:$1/health" >/dev/null 2>&1; }

status() {
    for s in "M2T2 8123" "FoundationStereo 1234"; do
        set -- $s
        up "$2" && echo "  $1 ($2)  up" || echo "  $1 ($2)  DOWN"
    done
    timeout 3 bash -c 'echo > /dev/tcp/192.168.68.132/5555' 2>/dev/null \
        && echo "  bamboo (192.168.68.132:5555)  up" \
        || echo "  bamboo (192.168.68.132:5555)  DOWN -- is the RT machine running its control node?"
}

start() {
    mkdir -p "$LOGS"
    up 8123 || { echo "starting M2T2..."
        (cd "$ROOT/droid/M2T2" && nohup pixi run server >"$LOGS/m2t2.log" 2>&1 &) ; }
    up 1234 || { echo "starting FoundationStereo..."
        (cd "$ROOT/droid/FoundationStereo" && nohup pixi run server >"$LOGS/fs.log" 2>&1 &) ; }

    echo "waiting for health (weights load takes ~30 s cold)..."
    for _ in $(seq 1 90); do up 8123 && up 1234 && break; sleep 2; done
    up 8123 && up 1234 || { echo "servers did not come up; see $LOGS/"; status; return 1; }

    echo "camera pre-flight:"
    $P python -m gwm_hardware.rs_preflight 2>/dev/null | grep -E "s/n|=>|PASS"
    echo "warming (absorbs the sm_120 PTX JIT):"
    $P python -m gwm_hardware.warm_servers --hand-serial 035422072950 2>/dev/null | grep -E "warm|healthy"
    echo; status
}

stop() { pkill -f "m2t2_server.py"; pkill -f "scripts/server.py"; pkill -f "rerun --port=9876"; echo "stopped"; }

case "${1:-status}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    run)    start && { echo; echo "launching tiptop-run -- type instructions at the prompt, 'exit' to quit";
                       cd "$ROOT/droid/tiptop" && set -a && . ./.env && set +a && exec pixi run tiptop-run "${@:2}"; } ;;
    *)      echo "usage: $0 {start|stop|status|run}"; exit 1 ;;
esac
