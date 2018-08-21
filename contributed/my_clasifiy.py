"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#Original author here https://github.com/davidsandberg/facenet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)

            if args.training_type == 'incremental':
                dataset, index = facenet.append_dataset(args.data_dir)
            else:
                dataset, index = facenet.get_dataset(args.data_dir)
                
            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset') 

            paths, labels = facenet.get_image_paths_and_labels(dataset, index)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            #raw_classifier_filename_exp = os.path.expanduser(args.classifier_filename+'.raw')
            model = SVC(kernel='linear', probability=True, verbose=True)
            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]
 
                if (args.training_type == 'incremental'): #add -----<--new
                    #with open(classifier_filename_exp, 'rb') as infile:
                    incremental_training(classifier_filename_exp, emb_array, labels, class_names)
                # Saving classifier model
                else:
                    #model = SVC(kernel='linear', probability=True, verbose=True)
                    #model = SGDClassifier(loss='log', verbose=True)
                    #model = MLPClassifier(verbose=True, solver='adam', hidden_layer_sizes=(500,250,125,100,75,50), learning_rate='constant', learning_rate_init=0.1, max_iter=10000, alpha=0.00001, tol=0.0)
                    #model.fit(emb_array, labels)
                    X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, test_size=0.25)
                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((X_train, y_train, class_names), outfile, protocol=pickle.HIGHEST_PROTOCOL)
                    
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

def incremental_training(filename, emb_array_new, labels_new, class_names_new):
    with open(filename, 'rb') as infile:
        emb_array_old, labels_old, class_names_old = pickle.load(infile)

    #for emb_array, labels, class_names in np.nditer(emb_array_new), labels_new, class_names_new:
    #    emb_array_old = np.append(emb_array_old, emb_array, axis=0)
    #    labels_old = labels_old.append(labels)
    #    class_names_old = class_names_old.append(class_names)

    emb_array_old = np.concatenate((emb_array_old, emb_array_new))
    #for emb_array in np.nditer(emb_array_new):
    #    emb_array_old = np.append(emb_array_old, emb_array, axis=0)

    #labels_new = np.array(labels_new) + max(labels_old) + 1                      # move numbers new list to old list end
    labels_old = np.concatenate((labels_old ,labels_new.tolist())).tolist()     #add new labels to old labels
    class_names_old = np.concatenate((class_names_old ,class_names_new))                          #append new class name to old class names
    
    with open(filename, 'wb') as outfile:
        pickle.dump((emb_array_old, labels_old, class_names_old), outfile, protocol=pickle.HIGHEST_PROTOCOL)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')

    parser.add_argument('--training_type', type=str, choices=['incremental', 'normal'], #------------<--new1
        help='Indicates whether you want to gradually train your data Or normal training.' + 
        'In the extra training you will need to provide the location of the features file.' +
        'informing you that it will be overwritten every time.', default='normal')

    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')

    parser.add_argument('--features_filename', # --------------<<----new2
        help='Features extracted file name as a pickle (.pkl) file. ' + 
        'path and name to file Features extracted and classes names and lable.', default='none')

    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_false')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=500)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=100)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=100)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


