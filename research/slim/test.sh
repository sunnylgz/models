python3 train_image_classifier.py \
  --train_dir='/tmp/silicon-models/vgg_16' \
  --dataset_name=silicon \
  --dataset_split_name=train \
  --dataset_dir='./silicon_tfrecord' \
  --model_name=vgg_16 \
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

