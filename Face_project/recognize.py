import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, request
from werkzeug.utils import secure_filename

KNOWN_FOLDER_Images = '/home/krishna/Machine learning/Face_project/known_faces_image'

class Recognize:
    def faceRecognize(self, fileName):
        # list of images in the folder
        images = []
        # list of names from the image by splitting the extension and getting the name only
        classNames = []
        # list of all the encoding(128-D) of the images in the folder
        encodeList = []

        # the number of images in dir
        myList = os.listdir(KNOWN_FOLDER_Images)
        # print(myList)

        # looping through all the images and getting the name without extension
        for id in myList:
            curImg = cv2.imread(f'{KNOWN_FOLDER_Images}/{id}')
            images.append(curImg)
            classNames.append(os.path.splitext(id)[0])
        # print(classNames)

        # looping through the list of images and returning the encoding(128-D) in a list
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            # print(len(encodeList))

            # From the video
            cap = cv2.VideoCapture(fileName)

            count = 0
            while True:
                success, frame = cap.read()
                # resizing the frame
                frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                # converting the frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detecting the location of face
                facesCurFrame = face_recognition.face_locations(frame)
                # Getting the encoding of frame
                encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)

                # match the encoding of frame with the encoding list of images above(encodeList)
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeList, encodeFace)
                    # getting the face distance after the comparison
                    faceDis = face_recognition.face_distance(encodeList, encodeFace)
                    # getting the minimum distance after the comparison
                    matchIndex = np.argmin(faceDis)

                    # if the match done then return the name
                    if matches[matchIndex]:
                        name = classNames[matchIndex]
                        count += 1
                    else:
                        return "Match not found"

                # break after 10 counts of matched found in the video
                if count == 10:
                    print("Predicted User:", name)
                    break

            # after the recognition delete the video file from the directory
            return name