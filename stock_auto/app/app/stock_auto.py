from datetime import datetime
import cv2
import re
import base64
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
import os,sys,datetime
import requests
import pandas
from graphpipe import remote
from matplotlib import pylab as plt

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = './result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    dt = datetime.datetime.fromtimestamp(os.stat('./result/result.png').st_mtime)
    disp_dt = dt.strftime('%Y-%m-%d %H:%M:%S')

    result = pandas.read_csv('../../learning//stock_auto_test.csv', usecols=[0,1,2,3,4,5], engine='python', names=('date','nissan','toyota','mazda','honda','subaru'))
#    df_table =  result.to_html(classes=["table", "table-condensed", "table-striped"],justify="center", escape=False)
    df_table =  result.to_html(classes=["table", "table-bordered", "table-hover"],justify="center")

    return render_template('index.html', img_url='./result/result.png', disp_dt=disp_dt, table=df_table)

@app.route('/result/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=False, host='0.0.0.0', port=8041)

