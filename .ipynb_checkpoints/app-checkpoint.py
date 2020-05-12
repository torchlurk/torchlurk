"""
/app.py:
Flask API to interact with the trained model. Cf README.md for further informations
"""
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
import getopt

short_options = "p:"
long_options = ["port="]
sys.path.insert(1,"./src")

# App definition
app = Flask(__name__,static_url_path="/static",static_folder="static",
            template_folder='templates',)

@app.route('/')
def home():
    return render_template('main.html')

if __name__ == "__main__":
    port = None
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)
    for curr_arg,curr_val in arguments:
        if curr_arg in ("-p","--port"):
            port = curr_val
    if port is None:
        port = 5000
    app.run(debug=True,host= '0.0.0.0',port=port)