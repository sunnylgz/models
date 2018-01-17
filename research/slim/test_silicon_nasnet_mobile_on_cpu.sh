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
DATASET_DIR=./silicon_tfrecord

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
#python3 download_and_convert_data.py \
#  --dataset_name=silicon \
#  --dataset_dir=${DATASET_DIR}

python3 eval_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/nasnet_mobile \
  --eval_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=silicon \
  --dataset_split_name=test \
  --model_name=nasnet_mobile \
  --eval_image_size=224 \
  --moving_average_decay=0.9999

return
# Fine-tune only the new layers for 1000 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=silicon \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nasnet_mobile \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/nasnet_mobile.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --clone_on_cpu=True \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --convert_input_to_rgb=True

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=silicon \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nasnet_mobile

# Fine-tune all the new layers for 500 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=silicon \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nasnet_mobile \
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
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nasnet_mobile
