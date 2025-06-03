#!/bin/bash

# Vars
HOST=jeanzay
REMOTE_PATH=/lustre/fswork/projects/rech/yfw/upp42qa/adaptive-defense/experiments/adadef
LOCAL_PATH=/Users/jeremie/Documents/01-Travail/01-Doctorat/adaptive-defense/experiments

# Logging
echo "rsync ${REMOTE_PATH} >> ${LOCAL_PATH} ..."
rsync \
    -zvar \
    --include='/*' \
    --exclude='mlruns/*' \
    --exclude='model.pt' \
    --exclude='predictions_train.pt' \
    $HOST:$REMOTE_PATH $LOCAL_PATH

#--include='**/stacks/* *_cfg.json *.csv' \
