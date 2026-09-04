# Sourced (bash -l) by the z-direct job scripts. Same env/cache handling as
# the sweep scripts plus the training paths.
source /work/vita/lanfeng/vlas/gwm-wiser/gwm_wiser/scripts/slurm/sweeps/common.sh
DATA_ROOT=${DATA_ROOT:-$REPO/wiser_dataset}
OUT_ROOT=${OUT_ROOT:-/work/vita/lanfeng/vlas/exp_results/gwm_zdirect}
ZRET_ROOT=${ZRET_ROOT:-$REPO/gwm_wiser_zdirect_ret}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
# Repo-scoped wandb identity (gitignored key file wins over ~/.netrc)
if [ -f "$REPO/.wandb_api_key.env" ]; then source "$REPO/.wandb_api_key.env"; fi
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
