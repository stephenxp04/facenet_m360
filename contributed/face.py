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
from sklearn.svm import SVC
import align.detect_face
import facenet


gpu_memory_fraction = 0.3
data_dir = '/home/stephenxp04/m360_face_images/my_dataset/train_aligned'
facenet_model_checkpoint = '/home/stephenxp04/m360_face_images/model_checkpoints/20180402-114759/20180402-114759.pb'
classifier_model = '/home/stephenxp04/m360_face_images/model_checkpoints/my_classifier_1.pkl'
debug = False
sess = None
graph = None

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
            face.confidence = self.identifier.confidence_level(face)
            #print(face.confidence)
            #face.name = self.identifier.identify(face)
        return faces


class Identifier(object):
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        #result = []
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            #result.append((self.class_names[best_class_indices[0]], best_class_probabilities[0]))
            return self.class_names[best_class_indices[0]]
            #return result

    def confidence_level(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            #result.append((self.class_names[best_class_indices[0]], best_class_probabilities[0]))
            return best_class_probabilities[0]

class Encoder(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.sess.as_default():
                facenet.load_model(facenet_model_checkpoint)
                print("Loaded model")
                with open('/tmp/log1', 'w') as f:
                    for node in self.graph.as_graph_def().node:
                        f.write(str(node.name + '\n'))

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def retrain_model(self):
        dataset = facenet.get_dataset(data_dir)

        np.random.seed(seed=666)
        # Check that there are at least one training image per class
        for cls in dataset:
            assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        with open('/tmp/log', 'w') as f:
            for node in self.graph.as_graph_def().node:
                f.write(str(node.name + '\n'))
        # Get input and output tensors
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * 90
            end_index = min((i + 1) * 90, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser(classifier_model)

        # if (args.mode=='TRAIN'):
        # Train classifier
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
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
