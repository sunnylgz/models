#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_nasnet_mobile_on_silicon.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=./pre_trained/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./silicon-models/nasnet_mobile

# Where the dataset is saved to.
DATASET_DIR=./imagenet

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir -p ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/nasnet_mobile/model.ckpt.data-00000-of-00001 ]; then
  wget https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz
  tar -xvf nasnet-a_mobile_04_10_2017.tar.gz
  mv nasnet_mobile.ckpt ${PRETRAINED_CHECKPOINT_DIR}/nasnet_mobile.ckpt
  rm nasnet-a_mobile_04_10_2017.tar.gz
fi

# Download the dataset
python3 download_and_convert_data.py \
  --dataset_name=imagenet \
  --dataset_dir=${DATASET_DIR}

python3 eval_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/nasnet_mobile \
  --eval_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=nasnet_mobile \
  --eval_image_size=224 \
  --moving_average_decay=0.9999

