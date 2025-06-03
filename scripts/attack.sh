#!/bin/bash
#SBATCH --account=yfw@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --job-name=attack-adadef
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks=4
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/logs/adadef/slurm-%A_%a.out.log
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/logs/adadef/slurm-%A_%a.out.log
#SBATCH --time=1:00:00

# NB: we export all variables so they are inherited by `srun` spawn environment
export SEED=0
export NUM_SHADOW=64

# Paths
export EXPERIMENT_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/experiments/adadef
export DATA_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/data
export REPO_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense
export LOG_PREFIX=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/logs/adadef/slurm-$SLURM_JOB_ID

# Environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export SRUN_UTILS_PATH="${REPO_DIR}/scripts/srun_utils.sh"

# Num CPU for the dataloader
export TORCH_DATALOADER_NUM_WORKER="4"
export OMP_NUM_THREADS=4

# Shared HPs
export NUM_CANARIES=500
export NUM_POISON=0
export POISON_TYPE="canary_duplicates_noisy"

export CANARY_TYPE_ALL=("ood" "clean")


for canary_type_idx in "${!CANARY_TYPE_ALL[@]}"; do
export CANARY_TYPE="${CANARY_TYPE_ALL[$canary_type_idx]}"
export EXPERIMENT="iclr_${CANARY_TYPE}"

echo "Attacking experiment ${EXPERIMENT}"

# We add `ntasks=1` to make sure that only one worker is spawn for each configuration
# We add `--overlap` to make sure that each task will get enough RAM
LOG_PATH=$LOG_PREFIX-$CANARY_TYPE.out.log
srun --ntasks=1 \
    --overlap  \
    --output $LOG_PATH \
    --error $LOG_PATH \
    $SRUN_UTILS_PATH python -u -m experiments.dpsgd \
        --experiment-dir "${EXPERIMENT_DIR}" \
        --experiment "${EXPERIMENT}" \
        --data-dir "${DATA_DIR}" \
        --seed "${SEED}" \
        --num-shadow "${NUM_SHADOW}" \
        --num-canaries "${NUM_CANARIES}" \
        --canary-type "${CANARY_TYPE}" \
        --num-poison "${NUM_POISON}" \
        --poison-type "${POISON_TYPE}" \
        attack &
done

wait
