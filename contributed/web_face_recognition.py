# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
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
import argparse
import sys
import time
import urllib2
import cv2
import face
import json

face_recognition = face.Recognition()
pic = '/home/m360/MachineLearning/my_dataset/train_aligned/Stephen/Stephen90.jpg'
# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name = "m360 Facial Recognition REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)


def add_overlays(faces):
    output = []

    if faces is not None:
        for f in faces:
            face_bb = f.bounding_box.astype(int)

            if f.name is not None:
                if f.confidence <= 0.70:
                    f.name = 'Unknown'

            person = Object()
            person.name = f.name
            person.score = f.confidence
            person.x = float(face_bb[0])
            person.y = float(face_bb[1])
            person.width = float(face_bb[2])
            person.height = float(face_bb[3])

            print(str(person.name) + ' : ' + str(person.score))
            output.append(person)

        outputJson = json.dumps([ob.__dict__ for ob in output])
        return outputJson
#             cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                         thickness=2, lineType=2)
#             cv2.putText(frame, (face.confidence * 100).astype(str) + "%", (face_bb[0], face_bb[1]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
#                         thickness=2, lineType=2)
# cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#             thickness=2, lineType=2)

def detect(image):
    faces = face_recognition.detect.find_faces(image)
    return faces

def recognize(image):
    #frame_interval = 2  # Number of frames after which to run face detection
    #fps_display_interval = 3  # seconds
    #frame_rate = 0
    #frame_count = 0

    #video_capture = cv2.VideoCapture(0)
    #start_time = time.time()

    #while True:
    # Capture frame-by-frame
    #ret, frame = video_capture.read()

    #if (frame_count % frame_interval) == 0:
    faces = face_recognition.identify(image)

    return add_overlays(faces)

def enrol():
    return face_recognition.encoder.retrain_model()
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()


# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--debug', action='store_true',
#                         help='Enable some debug outputs.')
#     return parser.parse_args(argv)
#
#
# if __name__ == '__main__':
# #    main(parse_arguments(sys.argv[1:]))
#      recognize(pic)

def debug():
    faces = face_recognition.identify(cv2.imread(pic, cv2.IMREAD_COLOR))
    print(str(add_overlays(faces)))

