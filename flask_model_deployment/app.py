##from flask import Flask,Request,url_for,render_template,Response
from flask import Flask,render_template,url_for,request,Response
import json
import os
import cv2
import math
import pafy
import random
import cv2
import numpy as np
import youtube_dl
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential ,model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.models import load_model


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET","POST"])

def predict():
    global RESULT
    RESULT=""
    global url
    url="https://www.youtube.com/watch?v=--4NLFGNfAs"
    global class_list
    class_list = ['push up', 'raising eye brows', 'celebrating', 'break dancing', 'climbing a rope', 'riding scooter', \
                  'calligraphy', 'clay pottery making', 'golf driving', 'eating icecream', 'cooking on campfire']
    json_file = open('C:\\Users\\DELL\\Desktop\\technocolabs\\10classes_CV_CNN\\final_version\\flask_deploy_the_model\\Data\\model552021.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global loaded_model
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:\\Users\\DELL\\Desktop\\technocolabs\\10classes_CV_CNN\\final_version\\flask_deploy_the_model\Data\\Model5.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print("Loaded model from disk")
    if request.method == "POST":
                ##print("in")
                url = request.form["urllink"]
                ##print(url)
                ##RESULT= real_predict(url)
                ##print(RESULT, list(RESULT))

    return render_template("result.html",prediction = "")
@app.route('/video_feed')
def video_feed():

    return Response(real_predict(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def real_predict(video_url='https://www.youtube.com/watch?v=CyDpR1RcHHo'):
    print("in")
    global cat_name
    cat_name = ''

    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)

    # get video formats available
    formats = info_dict.get('formats', None)
    predict_queue = []
    data = []
    for f in formats:

        # I want the lowest resolution, so I set resolution as 144p
        if f.get('format_note', None) == '360p':

            # get the video url
            url = f.get('url', None)

            # open url with opencv
            cap = cv2.VideoCapture(url)

            # check if url was opened
            if not cap.isOpened():
                print('video not opened')
                exit(-1)

            while True:
                # read frame

                ret, frame = cap.read()
                # check if frame is empty
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (64, 64))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                normalized_frame = resized_frame / 255
                prediction_result = loaded_model.predict(np.expand_dims(normalized_frame, axis=0))[0]
                predict_queue.append(prediction_result)


                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cat_name = max(set(data), key=data.count)
                    print("final result is ", cat_name)

                    ##return predicted_class_name
                    break

                if len(data) == 100:
                    cat_name = max(set(data), key=data.count)
                    print("final result is ", cat_name)
                    return cat_name


                # Assuring that the Deque is completely filled before starting the averaging process
                elif len(predict_queue) == 5:
                    # Converting Predicted Labels Probabilities Deque into Numpy array

                    predicted_labels_probabilities_np = np.array(predict_queue)

                    # Calculating Average of Predicted Labels Probabilities Column Wise
                    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis=0)

                    # Converting the predicted probabilities into labels by returning the index of the maximum value.
                    predicted_label = np.argmax(predicted_labels_probabilities_averaged)

                    # Accessing The Class Name using predicted label.
                    predicted_class_name = class_list[predicted_label]
                    data.append(predicted_class_name)
                    print(predicted_class_name)
                    # Overlaying Class Name Text Ontop of the Frame
                    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    predict_queue = []
                    ##cv2.imshow('frame', frame)
                    frame = cv2.imencode(".jpg",frame)[1].tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    # display frame




            # release VideoCapture
            cap.release()
    cv2.destroyAllWindows()

    return cat_name


if __name__ == "__main__":
    app.run(debug=True)