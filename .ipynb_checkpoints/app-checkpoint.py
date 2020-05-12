"""
/app.py:
Flask API to interact with the trained model. Cf README.md for further informations
"""
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
sys.path.insert(1,"./src")

# App definition
app = Flask(__name__,static_url_path="/static",static_folder="static",
            template_folder='templates',)

@app.route('/')
def home():
    return render_template('main.html')

if __name__ == "__main__":
    app.run(debug=True,host= '0.0.0.0',port=5000)