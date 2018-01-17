#! /bin/bash

DATASET_DIR=/tmp/flowers
TRAIN_DIR=/tmp/flowers-models/inception_v1
CHECKPOINT_PATH=/tmp/checkpoints/inception_v1.ckpt
python3 train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV1/Logits,InceptionV1/AuxLogits \
    --trainable_scopes=InceptionV1/Logits,InceptionV1/AuxLogits
