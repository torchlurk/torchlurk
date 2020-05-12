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
from model import run_model
from IPython.core.debugger import set_trace
from model import run_model,compute_pred

# App definition
app = Flask(__name__,template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/trainmodel',methods=['GET'])
def trainmodel():
    try:
        nb_epoch = int(request.args['nb_epoch'])
        run_model(nb_epochs=nb_epoch)
        resp = jsonify(success=True)
        resp.status_code = 200
        return resp
    except:
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)
    
@app.route('/predictGUI',methods=['POST'])
def predictGUI():
    file_name = request.form.get('filename')
    prediction = compute_pred(file_name)
    return render_template('index.html', 
                           emotion_pred='Emotion predicted: {}'.format(prediction['predicted']),
                           emotion_target='Real emotion: {}'.format(prediction['true_label']))

@app.route('/predict',methods=['POST'])
def predict():
    try:
        file_name = request.args.get('filename')
        prediction = compute_pred(file_name)
        resp = jsonify(prediction)
        resp.status_code = 200
        return resp
    except:
        #TODO: handle better the exceptions s.t. a clean messages displays in PostMan
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)
    
@app.route('/predictJSON',methods=['POST'])
def predictJSON():
    try:
        data = request.get_json(force=True)
        prediction = {val[0]:compute_pred(val[1]) for val in data.items()}
        resp = jsonify(prediction)
        resp.status_code = 200
        return resp
    except:
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True,host= '0.0.0.0',port=5000)