from datetime import datetime
import cv2
import re
import base64
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
import os,sys
import requests
from graphpipe import remote
from matplotlib import pylab as plt

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', img_url='../../learning/result.png')

if __name__ == '__main__':
    app.debug = True
    app.run(debug=False, host='0.0.0.0', port=8004)
