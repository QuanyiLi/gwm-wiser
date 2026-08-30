#!/usr/bin/env bash
# Replay the scene-6 candidate pools (pick: scene6_rev2, place: scene6_place_v2)
# and record frames / states / verdicts for the V-JEPA 2-AC selection eval.
set -u
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
PY=../droid-sim-evals/.venv/bin/python
$PY -u sim/replay_candidates.py --variant 0 --plans-dir ../gwm_integrate_doc/proposals/scene6_rev2 --out-dir runs/replay_pick > logs/replay_pick.log 2>&1
$PY -u sim/replay_candidates.py --variant 1 --plans-dir ../gwm_integrate_doc/proposals/scene6_place_v2 --out-dir runs/replay_place > logs/replay_place.log 2>&1
echo "ALL REPLAYS DONE" >> logs/replay_place.log
