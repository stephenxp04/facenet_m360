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
import threading
from threading import Thread
import face

face_recognition = None
face_det = None


def add_overlays(frame, faces, frame_rate):
    global face_det
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 0, 0), 1)
            if face.name is not None:
                if face.confidence <= 0.85:
                    face.name = ' '

                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
                cv2.putText(frame, (face.confidence * 100).astype(str) + "%", (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
                            thickness=2, lineType=2)
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    global face_recognition
    global face_det
    frame_interval = 2  # Number of frames after which to run face detection
    fps_display_interval = 3  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()
    video_capture.set(28, 0)
    #if args.debug:

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #img = cv2.imread(frame)

        if (frame_count % frame_interval) == 0:
            Thread(target=face_reg_wrapper, 
                args=(frame, )
            ).start()
            #print(face_det)
            #faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, face_det, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('r'):
            Thread(target=retrain_wrapper, verbose=True).start()
            continue

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def face_reg_wrapper(frame):
    global face_recognition
    global face_det
    #print(frame)
    #print("recognizing faces")
    #cv2.imshow("test", frame)
    face_det = face_recognition.identify(frame)
    #print(faces)
    #return face_recognition.identify(frame)

def retrain_wrapper():
    global face_recognition

    face_recognition.encoder.retrain_model(incremental=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
