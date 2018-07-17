#!/bin/sh

python /work/MachineLearning/facenet_m360/src/classifier.py TRAIN /work/MachineLearning/my_dataset/train_aligned/ /work/MachineLearning/model_checkpoints/20180402-114759/20180402-114759.pb /work/MachineLearning/model_checkpoints/my_classifier_1.pkl
