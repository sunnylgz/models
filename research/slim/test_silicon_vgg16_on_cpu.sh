#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an vgg_16 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_vgg_16_on_silicon.sh
set -e

# Where the pre-trained vgg_16 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/silicon-models/vgg_16

# Where the dataset is saved to.
DATASET_DIR=./silicon_tfrecord

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt ]; then
  wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  tar -xvf vgg_16_2016_08_28.tar.gz
  mv vgg_16.ckpt ${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt
  rm vgg_16_2016_08_28.tar.gz
fi

# Download the dataset
#python3 download_and_convert_data.py \
#  --dataset_name=silicon \
#  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=silicon \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt \
  --checkpoint_exclude_scopes=vgg_16/fc7,vgg_16/fc8 \
  --trainable_scopes=vgg_16/fc7,vgg_16/fc8 \
  --max_number_of_steps=1000 \
  --batch_size=1 \
  --clone_on_cpu=True \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --convert_input_to_rgb=True

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=silicon \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16

# Fine-tune all the new layers for 500 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=silicon \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --clone_on_cpu=True \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --convert_input_to_rgb=True

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=silicon \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16
