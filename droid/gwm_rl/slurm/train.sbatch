#!/usr/bin/env bash
#SBATCH --job-name=gwm_rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# One baseline run on one GPU:
#
#   sbatch slurm/train.sbatch 1                     # seed
#   sbatch slurm/train.sbatch 2 --wall-budget-s 900 # extra train.py args
#   bash   slurm/train.sbatch 1                     # same thing without Slurm
#
# ISAAC_PYTHON must point at a python with Isaac Lab 2.2 / Isaac Sim 5.0.
# GWM_RL_EXPERIMENT_ROOT redirects the run directory.
#
# A launch that has not printed "Time taken for simulation start" within
# START_TIMEOUT seconds is killed and retried (Kit can hang at PhysX start).
set -uo pipefail

SEED="${1:?seed}"
shift 1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GWM_RL="$(dirname "$HERE")"
PY="${ISAAC_PYTHON:-$GWM_RL/../droid-sim-evals/.venv/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
export GWM_RL_EXPERIMENT_ROOT="${GWM_RL_EXPERIMENT_ROOT:-$GWM_RL/experiments}"
NAME="${RUN_NAME:-pickbowl-s${SEED}}"
START_TIMEOUT="${START_TIMEOUT:-180}"
ATTEMPTS="${ATTEMPTS:-4}"
LOG="$GWM_RL_EXPERIMENT_ROOT/$NAME.log"

cd "$GWM_RL"
mkdir -p "$GWM_RL_EXPERIMENT_ROOT"
bash assets/fetch_assets.sh
"$PY" assets/make_assets.py >/dev/null

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== $NAME attempt $attempt $(date +%H:%M:%S)"
  # -u: stdout is block-buffered under redirection and Kit never flushes it.
  "$PY" -u train.py --headless --seed "$SEED" --no-time-suffix --exp-name "$NAME" \
    --final-eval-episodes 4 "$@" > "$LOG" 2>&1 &
  pid=$!
  t=0
  while kill -0 "$pid" 2>/dev/null && [ "$t" -lt "$START_TIMEOUT" ]; do
    grep -q "Time taken for simulation start" "$LOG" 2>/dev/null && break
    sleep 5; t=$((t + 5))
  done
  if kill -0 "$pid" 2>/dev/null && ! grep -q "Time taken for simulation start" "$LOG" 2>/dev/null; then
    echo "    startup hang after ${t}s — killing and retrying"
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    sleep 60
    continue
  fi
  wait "$pid"; rc=$?
  echo "exit $rc" >> "$LOG"
  grep -E "^  \[|first at|best |exploration cost|held out|^  success|Traceback" "$LOG" | tail -20
  grep -q '"final_eval"' "$GWM_RL_EXPERIMENT_ROOT/$NAME/evals.json" 2>/dev/null && exit 0
  echo "    run ended without a final evaluation (exit $rc); retrying"
  sleep 60
done
echo "=== $NAME FAILED after $ATTEMPTS attempts"
exit 1
