from flask import Flask, render_template, request, Response, send_from_directory, redirect, url_for, jsonify
import os
from PIL import Image
import json
import base64
import cv2
import numpy as np
#from src import classifier
import web_face_recognition

app = Flask(__name__)

save_path = str('/home/m360/MachineLearning/my_dataset/train_aligned/')

#for CORS
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
    return Response(open('./contributed/templates/index.html').read(), mimetype="text/html")


@app.route('/recognition')
def start():
    return Response(open('./contributed/templates/recognition.html').read(), mimetype="text/html")


@app.route('/enrol', methods=['POST'])
def enrol():
    try:
        if request.method == 'POST':
            print('POST /enrol success!')

        image_file = json.loads(request.data)
        name = image_file['id']

        if not os.path.exists(os.path.join(save_path+name)):
            os.mkdir(save_path+name)

        count = 0
        for images in image_file['data']:
            filename = name + str(count)

            img = base64.b64decode(images)
            img_array = np.fromstring(img, np.uint8)
            imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            faces = web_face_recognition.detect(imgdata)

            if len(faces) == 1:
                frame = faces[0].image
                rgb_frame = frame[:, :, ::-1]
                img = Image.fromarray(rgb_frame, "RGB")
                if img is not None:
                    img.save(os.path.join(save_path+name+'/'+filename+".jpg"))
                count += 1

                if count > 2000:
                    return redirect(url_for('getStatus', name=name))

        return redirect(url_for('getStatus', name=name))

    except Exception as e:
        print('POST /enrol error : %s' % e)
        return e


@app.route('/enrol/<name>')
def getStatus(name):
    try:
        return jsonify(name + ' enrolled ' + web_face_recognition.enrol())
        #return 'Test'

    except Exception as e:
        print('Enrolling failed : %s' % e)
        return e


@app.route('/recognition_result', methods=['POST'])
def face_recognition():
    try:
        if request.method == 'POST':
            print('POST /recognition_result success!')
            #print(request.data)

        # web_face_recognition.debug()
        image_file = json.loads(request.data)
        img = base64.b64decode(image_file['data'])
        img_array = np.fromstring(img, np.uint8)
        imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        boxes = web_face_recognition.recognize(imgdata)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)

