import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, request
from werkzeug.utils import secure_filename
from recognize import Recognize

app = Flask(__name__)
# Directory to save the image from the endpoint /train
KNOWN_FOLDER_Images = '/home/krishna/Machine learning/Face_project/known_faces_image'
# Directory to save the video from the endpoint /recognize
KNOWN_FOLDER_Videos = '/home/krishna/Machine learning/Face_project/known_faces_video'
app.config['KNOWN_FOLDER_Images'] = KNOWN_FOLDER_Images
app.config['KNOWN_FOLDER_Videos'] = KNOWN_FOLDER_Videos


# an endpoint to save the image
@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['KNOWN_FOLDER_Images'], filename))
    return "File Uploaded"


# an endpoint for facial recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    obj = Recognize()
    # an endpoint to recognize the face through video files of the person
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['KNOWN_FOLDER_Videos'], filename))
        result = obj.faceRecognize("known_faces_video/" + file.filename)
        os.remove("known_faces_video/" + file.filename)
    return result


# default running on http://127.0.0.1:5000/ by default
app.run()