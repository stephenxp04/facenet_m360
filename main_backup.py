from flask import Flask, render_template, request, Response, send_from_directory, redirect, url_for, jsonify
from flask_sslify import SSLify
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
#sslify = SSLify(app)
save_path = str('/work/MachineLearning/my_dataset/train_aligned/')
running = False
extProc = None
queue = []

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name = "Queue list"

    def toJSON(self):
        return json.dumps(self.__dict__)

#for CORS
# @app.before_first_request
# def activate_job():
#     def check_queue():
# 	global running
# 	global extProc
# 	global queue

# 	while True:
# 		if queue and extProc is not None:
# 			if sp.Popen.poll(extProc) is not None:
# 				queue.pop(0)
# 				print('start new task : ' + str(queue[0].name))
# 				extProc = sp.Popen('/work/MachineLearning/facenet_m360/retrain.sh', shell=True)
# 			else:
# 				print('still running old taks : ' + str(queue[0].name))
# 		else:
# 			running = False
# 			extProc = None
# 			print('no task')	
    			
# 		time.sleep(3)
#     thread = threading.Thread(target=check_queue)
#     thread.start()

#@app.before_request
#def before_request():
#    if not request.url.startswith('http://'):
#	url = request.url.replace('http://', 'https://', 1)
#	code = 301
#	return redirect(url, code=code)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/')
def index():
    return Response(os.getcwd())


@app.route('/app')
def remote():
    return Response(open('/work/MachineLearning/facenet_m360/contributed/templates/index.html').read(), mimetype="text/html")


@app.route('/recognition')
def start():
    return Response(open('/work/MachineLearning/facenet_m360/contributed/templates/recognition.html').read(), mimetype="text/html")


@app.route('/enrol', methods=['POST'])
def enrol():
    global queue
    global running
    global extProc
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
            #faces = web_face_recognition.detect(imgdata)

            #if len(faces) == 1:
            #    frame = faces[0].image
            rgb_frame = imgdata[:, :, ::-1]
            img = Image.fromarray(rgb_frame, "RGB")
            if img is not None:
                img.save(os.path.join(save_path+name+'/'+filename+".jpg"))
            count += 1

	    person = Object()
	    person.name = name
	    queue.append(person)

        if running is False:
            running = True
            web_face_recognition.enrol(incremental=True)

        return redirect(url_for('getStatus', name=name))

    except Exception as e:
        print('POST /enrol error : %s' % e)
        return e


@app.route('/getStatus')
def getStatus():
    try:
	global running
	global extProc
	global queue
	
	req = str(request.args.get('name'))

	return json.dumps([ob.__dict__ for ob in queue])

    except Exception as e:
        print('Enrolling failed : %s' % e)
        return e


@app.route('/recognition_result', methods=['POST'])
def face_recognition():
    try:
        if request.method == 'POST':
            start = time.time()    	
            print('POST /recognition_result success!')

        # web_face_recognition.debug()
        image_file = json.loads(request.data)
        img = base64.b64decode(image_file['data'])
        img_array = np.fromstring(img, np.uint8)
        imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        boxes = web_face_recognition.recognize(imgdata)
	end = time.time()
	print(end - start)
        return boxes

    except Exception as e:
        print('Recognition failed : %s' % e)
        return e


@app.route('/loading.gif')
def loading():
    return send_from_directory(os.path.join(app.root_path, 'templates'),
                               'loading.gif', mimetype='image/gif')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# def start_runner():
#     def start_loop():
#         not_started = True
#         while not_started:
#             print('In start loop')
#             try:
#                 r = requests.get('http://localhost:8081/app')
#                 #r = requests.get('https://ml.deekie.com/enrol/app')
#                 if r.status_code == 200:
#                     print('Server started, quiting start_loop')
#                     not_started = False
#                 print(r.status_code)
#             except:
#                 print('Server not yet started')
#             time.sleep(2)

#     print('Started runner')
#     thread = threading.Thread(target=start_loop)
#     thread.start()

if __name__ == '__main__':
    #start_runner()
    app.run(host="localhost", port=8081, threaded=True)
    #app.run(host='0.0.0.0', port=8081, threaded=True, debug=True, ssl_context="adhoc")
    

