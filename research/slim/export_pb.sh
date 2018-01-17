#python3 export_inference_graph.py --dataset_name=silicon --dataset_dir=silicon_tfrecord/ --model_name=inception_v3 --output_file=silicon-models/inception_v3_2/inceptionv3.pb

#python3 /usr/local/lib/python3.4/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=silicon-models/inception_v3_2/inceptionv3.pb --input_checkpoint=silicon-models/inception_v3_2/model.ckpt-20000 --input_binary=true --output_graph=silicon-models/inception_v3_2/frozen_inceptionv3.pb --output_node_names=InceptionV3/Predictions/Reshape_1

python3 /usr/local/lib/python3.4/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=silicon-models/inception_v3_2/inceptionv3.pb --input_checkpoint=silicon-models/inception_v3_2/all/model.ckpt-10000 --input_binary=true --output_graph=silicon-models/inception_v3_2/all/frozen_inceptionv3.pb --output_node_names=InceptionV3/Predictions/Reshape_1
python3 /usr/local/lib/python3.4/dist-packages/tensorflow/python/tools/optimize_for_inference.py --input silicon-models/inception_v3_2/all/frozen_inceptionv3.pb --output silicon-models/inception_v3_2/all/frozen_inceptionv3_optimized.pb --input_names "input" --output_names "InceptionV3/Predictions/Reshape_1" --frozen_graph=true

python3 export_inference_graph.py --dataset_name=silicon --dataset_dir=silicon_tfrecord/ --model_name=inception_resnet_v2 --output silicon-models/inception_resnet_v2/inception_resnet_v2.pb

python3 /usr/local/lib/python3.4/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=silicon-models/inception_resnet_v2/inception_resnet_v2.pb --input_checkpoint=silicon-models/inception_resnet_v2/all/model.ckpt-15000 --input_binary=true --output_graph=silicon-models/inception_resnet_v2/all/frozen_inception_resnet_v2.pb --output_node_names=InceptionResnetV2/Logits/Predictions
