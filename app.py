from flask import Flask, render_template, request
from scipy.misc import imread, imresize
import numpy as np
import re
import sys
import os
import base64

from model.load import init

app = Flask(__name__)

global nn_model, graph
nn_model, graph = init()

# aux functions
# decode img from base 64 to raw
def convertImage(imgdata):
    imgstr = re.search(r'base64, (.*)', str(imgdata)).group(1)
    
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict/', methods = ['GET', 'POST'])

def predict():
    imgdata = request.get_data()
    convertImage(imgdata)
    image = imread('output.png', mode = 'L')
    image = imresize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)

    with graph.as_default():
        out = nn_model.predict(image)
        print(out)
        response = np.argmax(out, axis = 1)
        print(response)

    return str(response[0])

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

