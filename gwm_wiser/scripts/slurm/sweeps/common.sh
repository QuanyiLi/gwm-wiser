# Sourced (bash -l) by every sweep job script in this directory.
# Sets paths, activates the WISER conda env and keeps every cache off /home.
REPO=${REPO:-/work/vita/lanfeng/vlas/gwm-wiser}
CONDA_BASE=${CONDA_BASE:-/home/lfeng/miniconda3}
CONDA_ENV=${CONDA_ENV:-/work/vita/lanfeng/conda_env/gwm-wiser}
EMBEDDER=${EMBEDDER:-/work/vita/lanfeng/vlas/Qwen3-VL-Embedding-8B}
DATASET=${DATASET:-$REPO/wiser_dataset/no_noise_demo_1_round}
RET_ROOT=${RET_ROOT:-$REPO/gwm_wiser_exp_ret}
EVAL_PY=$REPO/gwm_wiser/scripts/gwm_eval.py

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# slurm 24.11 no longer propagates --cpus-per-task to srun
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export PYTHONUNBUFFERED=1
export SVT_LOG=0
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/vita/lanfeng/.huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet

# Node-local scratch for the lerobot/datasets arrow cache. gwm_eval.py sets
# HF_DATASETS_CACHE only after `import datasets`, which is a no-op, so it has
# to be exported here before python starts.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/tmp/${SLURM_JOB_ID}" ]; then
    JOB_TMP=/tmp/${SLURM_JOB_ID}/sweep
else
    JOB_TMP=/tmp/sweep_${SLURM_JOB_ID:-$$}
fi
mkdir -p "$JOB_TMP"
export TMPDIR=$JOB_TMP
export HF_DATASETS_CACHE=$JOB_TMP/hf_datasets
cleanup_job_tmp() { rm -rf "$JOB_TMP"; }

# wait_with_watchdog <pid> <logfile> <max_silent_seconds>
# Returns the process exit code, or 124 if the log stopped growing (SAPIEN
# init hangs forever on nodes with a broken Vulkan ICD).
wait_with_watchdog() {
    local pid=$1 log=$2 stall=$3
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        local mtime now age
        mtime=$(stat -c %Y "$log" 2>/dev/null || date +%s)
        now=$(date +%s)
        age=$((now - mtime))
        if [ "$age" -gt "$stall" ]; then
            echo "[watchdog] $log silent for ${age}s (> ${stall}s) on $(hostname); killing $pid"
            kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
    done
    wait "$pid"
    return $?
}

summarize_metrics() {  # <episode_metrics.json>
    python - "$1" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
keys = ["success_at_end_mean", "is_grasped_mean", "tcp_near_goal_mean", "rollout_time"]
print("  " + ", ".join(f"{k}={s[k]:.3f}" for k in keys if k in s))
PY
}

# run_eval <result_dir> <split> <start> <end> <replan N> <horizon H> <keyframes K> <logfile> [extra args]
# GT-MPC (oracle, --use_gt) evaluation of configs [start, end) with the given
# planning hyper-parameters. Resumable: gwm_eval.py skips configs that already
# have episode_metrics.json.
run_eval() {
    local result_dir=$1 split=$2 start=$3 end=$4 N=$5 H=$6 K=$7 log=$8; shift 8
    mkdir -p "$result_dir" "$RET_ROOT/.cwd"
    cat > "$result_dir/run_config_${start}_${end}_${split}.json" <<JSON
{"script": "gwm_wiser/scripts/gwm_eval.py --use_gt", "planner": "RetrievalBasedPlanner (GT-MPC / oracle)",
 "embedder_model_path": "$EMBEDDER", "dataset_root": "$DATASET",
 "replan_horizon": $N, "num_future_frames": $H, "video_frame_subsample": $K,
 "k": 12, "num_env": 12, "eval_rounds": 1, "split": "$split",
 "start_subset": $start, "end_subset": $end,
 "slurm_job_id": "${SLURM_JOB_ID:-}", "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
 "hostname": "$(hostname)", "git_commit": "$(git -C $REPO rev-parse HEAD)", "date": "$(date -Is)"}
JSON
    # gwm_eval.py creates a ManiSkill record dir "test" relative to the cwd.
    cd "$RET_ROOT/.cwd"
    python -u "$EVAL_PY" --use_gt --dataset_root "$DATASET" \
        --embedder_model_path "$EMBEDDER" --result_dir "$result_dir" \
        --start_subset "$start" --end_subset "$end" --split "$split" --eval_rounds 1 \
        --replan_horizon "$N" --num_future_frames "$H" --video_frame_subsample "$K" \
        "$@" > "$log" 2>&1 &
    wait_with_watchdog $! "$log" 1800
    local rc=$?
    echo "== $(date) eval [$start,$end) split=$split N=$N H=$H K=$K exit=$rc"
    for i in $(seq "$start" $((end - 1))); do
        local f=$result_dir/config_${i}_${split}/episode_metrics.json
        if [ -f "$f" ]; then printf "config_%s_%s:" "$i" "$split"; summarize_metrics "$f"; else echo "config_${i}_${split}: MISSING"; fi
    done
    return $rc
}
