from flask import Flask, render_template, request, Response, send_from_directory
from enrol_face import Enrol
import os
from PIL import Image
import json
import base64
from subprocess import call

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
    return Response(recognition(), 'Facial Recognition - Profile Enrolment', open('./contributed/templates/enrol.html').read(), mimetype="text/html")


@app.route('/app')
def remote():
    return Response(open('./contributed/templates/index.html').read(), mimetype="text/html")


@app.route('/enrol', methods=['POST'])
def enrol():
    try:
        if request.method == 'POST':
            print('POST /enrol success!')
            #print(request.data)

        #string = json.dumps(request.data)
        image_file = json.loads(request.data)
        name = image_file['id']

        if not os.path.exists(os.path.join(save_path+name)):
            os.mkdir(save_path+name)

        count = 0
        for images in image_file['data']:
            imgdata = base64.b64decode(images)
            filename = name + str(count)
            with open(os.path.join(save_path+name+'/'+filename+'.jpg'), 'wb') as f:
                f.write(imgdata)
            count = count + 1

        call('./retrain.sh')

        return json.dumps(image_file)

    except Exception as e:
        print('POST /enrol error : %s' % e)
        return e
#@app.route('/recognition', methods=['POST'])
#def recognition():
#    return True


#@app.route('/favicon.ico')
#def favicon():
#    return send_from_directory(os.path.join(app.root_path, 'static'),
#                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='192.168.0.41', debug=True)

