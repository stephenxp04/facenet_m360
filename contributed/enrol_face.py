# coding=utf-8
"""Performs face enrolling in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
import argparse
import sys
import time
import os
import cv2

import face
from PIL import Image


class Enrol(object):
    def main(args):
        frame_interval = 3  # Number of frames after which to run face detection
        fps_display_interval = 5  # seconds
        frame_rate = 0
        frame_count = 0
        count = 0
        save_path = str('/work/MachineLearning/my_dataset/train_aligned/' + args.name)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        print("Saving images into " + save_path)
        video_capture = cv2.VideoCapture(0)
        face_detection = face.Detection()
    #   face_recognition = face.Recognition()
        start_time = time.time()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if (frame_count % frame_interval) == 0:
                faces = face_detection.find_faces(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

    #        add_overlays(frame, faces, frame_rate)

            frame_count += 1
            if len(faces) == 1:
                frame = faces[0].image
                cv2.imshow('Enrolling', frame)
                cv2.setWindowTitle('Enrolling', str(args.name) + " " + str(count+1))
                #cv2.putText(faces[0].image, 'Image: ' + str(frame_count+1), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #            (255, 0, 0), thickness=2, lineType=2)
                rgb_frame = frame[:, :, ::-1]
                img = Image.fromarray(rgb_frame, "RGB")
                if img is not None:
                    img.save(os.path.join(save_path+"/"+str(count)+".jpg"))
                count += 1

                #if frame_count > 100:
                #   break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

    def parse_arguments(argv):
        parser = argparse.ArgumentParser()

        parser.add_argument('--name', type=str, required=True,
                            help='Name of input person')
        return parser.parse_args(argv)

    if __name__ == '__main__':
        main(parse_arguments(sys.argv[1:]))