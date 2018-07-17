# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
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

import pickle
import os
import math
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf
from sklearn.neighbors import KNeighborsClassifier
from multiprocess import Pool, cpu_count
import align.detect_face
import facenet
from tqdm import tqdm
from timeit import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from joblib import Parallel, delayed

gpu_memory_fraction = 0.5
data_dir = '/work/MachineLearning/my_dataset/train_aligned'
facenet_model_checkpoint = '/work/MachineLearning/model_checkpoints/20180402-114759/20180402-114759.pb'
#classifier_model = '/work/MachineLearning/model_checkpoints/incremental_test.pkl'
classifier_model = '/work/MachineLearning/model_checkpoints/incremental_raw.pkl'

debug = False
sess = None
graph = None
model = SVC(kernel='linear', probability=True)
labels = None
class_names = None
emb_array = None

class Face(object):
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.confidence = None

class Recognition(object):
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)
#
        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)
            #face.confidence = 1.0
            face.confidence = self.identifier.confidence_level(face)
        return faces


class Identifier(object):
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.emb_array, self.labels, self.class_names = pickle.load(infile)
        self.build_model()

    def build_model(self):
        global labels
        global emb_array
        global class_names

        emb_array = self.emb_array
        labels = self.labels
        class_names = self.class_names
        #emb_array = np.concatenate((emb_array, self.emb_array))
        #labels = np.concatenate((labels , self.labels.tolist())).tolist()     #add new labels to old labels
        #class_names = np.concatenate((class_names , self.class_names))                          #append new class name to old class names

        X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, test_size=0.2)
        #self.model = SGDClassifier(loss='log', verbose=True, n_jobs=-1, n_iter=1000, alpha=1e-5, 
        #    tol=None, shuffle=True, random_state=100, penalty='l2')
       #self.model = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
        print('Start building model')
        start = time.time()
        #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #self.model = GridSearchCV(SVC(kernel='rbf', cache_size=2048, probability=True), param_grid, n_jobs=-1)
        model.fit(X_train, y_train)
        #self.model.fit(X_train, y_train)
        end = time.time()
        print ("Fit Time: {0:4f}s".format(end - start))
        print('Build model done')

    def identify(self, face):
        global model
        global class_names
        if face.embedding is not None:
            predictions = model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            return class_names[best_class_indices[0]]
   
    def confidence_level(self, face):
        global model
        if face.embedding is not None:
            predictions = model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            return best_class_probabilities[0]

class Encoder(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.sess.as_default():
                facenet.load_model(facenet_model_checkpoint)
                print("Loaded model")

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def incremental_training(self, emb_array_new, labels_new, class_names_new):
        global labels
        global emb_array
        global class_names
        emb_array = np.concatenate((emb_array, emb_array_new))
        labels = np.concatenate((labels ,labels_new))     #add new labels to old labels
        class_names = np.concatenate((class_names ,class_names_new))                          #append new class name to old class names

    def retrain_model(self, incremental):
        global labels
        global emb_array
        global class_names
        if incremental is True:
            dataset, append_index = facenet.append_dataset(data_dir)
            paths, self.append_labels = facenet.get_image_paths_and_labels(dataset, append_index)
            self.append_class_names = [cls.name.replace('_', ' ') for cls in dataset]

        else:
            dataset, append_index = facenet.get_dataset(data_dir)
            paths, labels = facenet.get_image_paths_and_labels(dataset, append_index)
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

        np.random.seed(seed=666)
        # Check that there are at least one training image per class
        for cls in dataset:
            assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')
        
        # Create a list of class names

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        if incremental is True:
            print("new people added: ")
            print(self.append_class_names)
 
        # Get input and output tensors
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
        self.append_emb_array = np.zeros((nrof_images, embedding_size))

        for i in tqdm(range(nrof_batches_per_epoch)):
            start_index = i * 90
            end_index = min((i + 1) * 90, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            self.append_emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser(classifier_model)
        print('Training classifier')
     
        if incremental is True:
            self.incremental_training(self.append_emb_array, self.append_labels, self.append_class_names)
  
        X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, test_size=0.25)
        #self.model = SGDClassifier(loss='log', verbose=True, n_jobs=-1, n_iter=1000, alpha=1e-5, 
        #    tol=None, shuffle=True, random_state=100, penalty='l2')
        #self.model = SVC(kernel='rbf', probability=True, verbose=True, cache_size=1024)
        #self.model = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
        print('Start building model')
        start = time.time()
        #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #self.model = GridSearchCV(SVC(kernel='rbf', cache_size=2048, probability=True), param_grid, n_jobs=-1)
        model.fit(X_train, y_train)
        #self.model.fit(X_train, y_train)
        end = time.time()
        print ("Fit Time: {0:4f}s".format(end - start))
        print('Build model done')
       
        if incremental is False:
            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((emb_array, labels, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)

        return 'Success'

class Detection(object):
    # face detection parameters
    minsize = 30  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces
