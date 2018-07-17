from flask import Flask, render_template, request, Response, send_from_directory, redirect, url_for, jsonify
from flask_sslify import SSLify
from flask_cors import CORS
import os
from PIL import Image
import json
import base64
import cv2
import numpy as np
#from src import classifier
import web_face_recognition
import time
import subprocess as sp
import requests
import threading
from threading import Thread

app = Flask(__name__)
sslify = SSLify(app)
CORS(app)
save_path = str('/work/MachineLearning/my_dataset/train_aligned/')

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name = "Queue list"

    def toJSON(self):
        return json.dumps(self.__dict__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.route('/')
def index():
    return Response(os.getcwd())

@app.route('/recognition')
def start():
    return Response(open('/work/MachineLearning/facenet_m360/contributed/templates/single_shot_recognition.html').read(), mimetype="text/html")

@app.route('/enrol', methods=['POST'])
def enrol():
    try:
        if request.method == 'POST':
            print('POST /enrol success!')

        image_file = json.loads(request.data)
        name = str(image_file['id'])

        if not os.path.exists(os.path.join(save_path+name)):
            os.mkdir(save_path+name)

        count = 0
        for images in image_file['data']:
            filename = name + str(count)
            img = base64.b64decode(images)
            img_array = np.fromstring(img, np.uint8)
            imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            rgb_frame = imgdata[:, :, ::-1]
            img = Image.fromarray(rgb_frame, "RGB")
            if img is not None:
                img.save(os.path.join(save_path+name+'/'+filename+".jpg"))
            count += 1

        Thread(target=web_face_recognition.enrol, args=[name, incremental=True]).start()
        #web_face_recognition.enrol(name, incremental=True)

        return 

    except Exception as e:
        print('POST /enrol error : %s' % e)
        return e


@app.route('/getStatus', methods=['GET', 'POST'])
def getStatus():
    try:
        output = []
        queue = web_face_recognition.get_status()

        return queue

    except Exception as e:
        print('Check status failed : %s' % e)
        return e


@app.route('/recognition_result', methods=['POST'])
def face_recognition():
    try:
        image_file = json.loads(request.data)
        img = base64.b64decode(image_file['data'])
        img_array = np.fromstring(img, np.uint8)
        imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        boxes = web_face_recognition.recognize(imgdata)

        return boxes

    except Exception as e:
        print('Recognition failed : %s' % e)
        return e

if __name__ == '__main__':
    #start_runner()
    app.run(host="0.0.0.0", port=8081, threaded=True, ssl_context='adhoc')
    #app.run(host='0.0.0.0', port=8081, threaded=True, debug=True, ssl_context="adhoc")
    

