# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:29:52 2021

@author: Muadh
"""

from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('simple_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    pred_price = model.predict(final_features)
    output = round(pred_price[0], 2)
    return render_template('page.html', prediction_text='Iris flower class is {}'
                           .format(output))

if __name__ == '__main__':
    app.run(debug=True)