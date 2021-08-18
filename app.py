# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 08:23:55 2021

@author: dilip-k
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)
rf_model = pickle.load(open('RF_model.pkl', 'rb'))
Band_encoder = pickle.load(open('Band_encoder.pkl', 'rb'))
Cat_encoder = pickle.load(open('Cat_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    subband = request.form['SubBand']
    skill = request.form['Skill']
    exp = int(request.form['Experience'])
    array = np.array([[subband,skill,exp]])
    
    index_values=[0]
    column_values=['Subband','Category','Experience']
    
    X_newTest = pd.DataFrame(data=array,
                             index=index_values,
                             columns = column_values)
    
    X_newTest = Band_encoder.transform(X_newTest)
    X_newTest = Cat_encoder.transform(X_newTest)
    
    #print(X_newTest)
    X_newTest = scaler.transform(X_newTest)
    
    predict = rf_model.predict(X_newTest)
    
    output = predict[0].astype(int)
    
    return render_template('index.html', prediction_text='Resource Cost = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    