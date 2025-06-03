#!/bin/bash
#SBATCH --account=yfw@a100
#SBATCH --array=0-64
#SBATCH --job-name=adadef
#SBATCH -C a100
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-gpu=8
#SBATCH --ntasks=2
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/logs/adadef/slurm-%A_%a.out.log
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/logs/adadef/slurm-%A_%a.out.log
#SBATCH --time=4:00:00
# Train on 2 canary types x 64 shadow models per setting = 256 models

# NB: we export all variables so they are inherited by `srun` spawn environment
export SEED=0
export NUM_SHADOW=64
export NUM_MODEL_PER_GPU=2

# Paths
export EXPERIMENT_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/experiments/adadef
export DATA_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/data
export REPO_DIR=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense

# Environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export SRUN_UTILS_PATH="${REPO_DIR}/scripts/srun_utils.sh"

# Shared HPs
export AUGMULT_FACTOR="8"

# Num CPU for the dataloader
export TORCH_DATALOADER_NUM_WORKER="2"
export OMP_NUM_THREADS=2

# Training Hpars
# export NUM_EPOCHS="100"
export NUM_EPOCHS="300"
export LEANING_RATE=".1"
export MOMENTUM="0.9"
export WEIGHT_DECAY="0.0001"
export BATCH_SIZE="64"

# Defense Hpars -- important
# export BASE_RELU_SLOPE="4"
# export SLOPE_EPOCH_INCREASE=".05"
export BASE_RELU_SLOPE="0"
export SLOPE_EPOCH_INCREASE="0"

# Defense Hpars -- miscelaneous
export ALPHA_NORM="0.0001"
export RELU_QUANTILE_OFFSET="0"
export LOSS_TRACKER_DECAY_THRESHOLD="0.95"
export LOSS_TRACKER_NUM_SAMPLES="500"

export NUM_CANARIES=500
export NUM_POISON=0
export POISON_TYPE="canary_duplicates_noisy"
export CANARY_TYPE_ALL=("ood" "clean")


for ((LOCAL_MODEL_IDX=0; LOCAL_MODEL_IDX<NUM_MODEL_PER_GPU; LOCAL_MODEL_IDX++))
do
    # Global idx of the model - Checking that there is no more than 256 models
    export GLOBAL_MODEL_IDX=$((SLURM_ARRAY_TASK_ID * NUM_MODEL_PER_GPU + LOCAL_MODEL_IDX))

    if (( GLOBAL_MODEL_IDX > 127 ))
    then
        break
    fi

    # Local HPs
    export CANARY_TYPE_IDX=$((GLOBAL_MODEL_IDX / NUM_SHADOW))
    export SHADOW_MODEL_IDX=$((GLOBAL_MODEL_IDX % NUM_SHADOW))
    export CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"
    export EXPERIMENT="iclr_${CANARY_TYPE}"

    echo "Running model ID ${GLOBAL_MODEL_IDX} on task ID ${SLURM_ARRAY_TASK_ID}"
    echo "Experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
    echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

    # We add `ntasks=1` to make sure that only one worker is spawn for each configuration
    # We add `overlap` to make sure the GPU will be shared for all tasks
    srun --ntasks=1 --overlap $SRUN_UTILS_PATH python -u -m experiments.adadef \
        --experiment-dir "${EXPERIMENT_DIR}" \
        --experiment "${EXPERIMENT}" \
        --data-dir "${DATA_DIR}" \
        --seed "${SEED}" \
        --num-shadow "${NUM_SHADOW}" \
        --num-canaries "${NUM_CANARIES}" \
        --canary-type "${CANARY_TYPE}" \
        --num-poison "${NUM_POISON}" \
        --poison-type "${POISON_TYPE}" \
        train \
            --shadow-model-idx "${SHADOW_MODEL_IDX}" \
            --augmult-factor "${AUGMULT_FACTOR}" \
            --num-epochs "${NUM_EPOCHS}" \
            --learning-rate "${LEANING_RATE}" \
            --momentum "${MOMENTUM}" \
            --weight-decay "${WEIGHT_DECAY}" \
            --batch-size "${BATCH_SIZE}" \
            --base-relu-slope "${BASE_RELU_SLOPE}" \
            --relu-quantile-offset "${RELU_QUANTILE_OFFSET}" \
            --slope-epoch-increase "${SLOPE_EPOCH_INCREASE}" \
            --alpha-norm "${ALPHA_NORM}" \
            --loss-tracker-decay-threshold "${LOSS_TRACKER_DECAY_THRESHOLD}" \
            --loss-tracker-num-samples "${LOSS_TRACKER_NUM_SAMPLES}" &

done

wait
