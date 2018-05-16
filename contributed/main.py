from flask import Flask, render_template, request, Response, send_from_directory
from enrol_face import Enrol
import os
from PIL import Image
import json
import base64
from subprocess import call
import face
import cv2
import numpy as np

app = Flask(__name__)
enrol = None
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
    #return render_template('index.html')
    return Response('Facial Recognition - Profile Enrolment')


@app.route('/app')
def remote():
    return Response(open('./contributed/templates/index.html').read(), mimetype="text/html")


@app.route('/enrol', methods=['POST'])
def enrol():
    try:
        if request.method == 'POST':
            print('POST /enrol success!')
            #print(request.data)

        face_detection = face.Detection()
        #string = json.dumps(request.data)
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
            #if not os.path.exists(os.path.join('/tmp/'+name)):
            #    os.mkdir(os.path.join('/tmp/'+name))
            #with open(os.path.join('/tmp/'+name+'/'+filename+'.jpg'), 'wb') as f:
            #    f.write(img)

            #img = cv2.imread(f.name, cv2.IMREAD_COLOR)
            faces = face_detection.find_faces(imgdata)

            if len(faces) == 1:
                frame = faces[0].image
                #cv2.imshow('Enrolling', frame)
                #cv2.setWindowTitle('Enrolling', str(args.name) + " " + str(count+1))
                #cv2.putText(faces[0].image, 'Image: ' + str(frame_count+1), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #            (255, 0, 0), thickness=2, lineType=2)
                rgb_frame = frame[:, :, ::-1]
                img = Image.fromarray(rgb_frame, "RGB")
                if img is not None:
                    img.save(os.path.join(save_path+name+'/'+filename+".jpg"))
                count += 1

        #call('./retrain.sh')

        return "enrol done"

    except Exception as e:
        print('POST /enrol error : %s' % e)
        return e

@app.route('/recognition', methods=['POST'])
def recognition():
    return Response('Nothing')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    app.run(host='0.0.0.0', debug=True, threaded=True)

