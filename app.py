from __future__ import division, print_function
from flask import Flask, request

# coding=utf-8
import sys
import os
import glob
import re
import pandas as pd
import numpy as np 


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
#for keras and model loading
#from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


#start of app
app = Flask(__name__)

model_path ='model_full_inceptionv3.h5'
model = load_model(model_path)


def predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size =(224, 224))
    x = image.img_to_array(img)

    x = x/225
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Person is normal"
    elif preds==1:
        preds="Person have pneumonia"
    else:
        print('error')
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict(file_path, model)
        result=preds
        return result
    return None




if __name__ == '__main__':
    app.run(debug = True)