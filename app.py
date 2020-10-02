"""
/app.py:
Flask API to interact with the trained model. Cf README.md for further informations
"""
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template,send_from_directory
import pickle
import sys
from pathlib import Path
import getopt
from time import sleep
from werkzeug.serving import make_server
import os


local = Path(__file__).parent
html_path = local/'main.html'

short_options = "p:"
long_options = ["port="]
sys.path.insert(1,"./src")

# App definition
app = Flask(__name__,static_url_path="/static",static_folder="static",
            template_folder='templates',)

@app.route('/')
def home():
    return render_template('main.html')
# Custom static dlabels_path="./bigdata/imagenet-mini/labels_imagenet.txt",ata

@app.route('/<path:filename>')
def wellKnownRoute(filename):
    # adapth this path when working from outside the directory
    # copy templates and static with you
    return send_from_directory(os.getcwd(),filename)

def start(port=5000):
    app.run(debug=True,host= '0.0.0.0',port=port,use_reloader=False)
if __name__ == "__main__":
    start()
    
