import pickle
from flask import Flask,request, render_template,app,jsonify
import numpy as np
import pandas as pd


app = Flask(__name__)
# Loading the model
model = pickle.load(open('classifire.pkl','rb'))
# Loading scaling
scaling = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=="POST":
        data = [int(x) for x in request.form.values()]
        new_data = scaling.transform(np.array(data).reshape(1,-1))
        output = model.predict(new_data)[0]
        return render_template("output.html",prediction = output)

if __name__=="__main__":
    app.run(debug=True)