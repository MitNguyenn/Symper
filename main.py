from flask import Flask, request, jsonify
import pandas as pd
from models.NaiveBayes import train as train_naive_bayes
from models.LinearRegression import train as train_linear_regression
from models.LogisticRegression import train as train_logistic_regression
import uuid

app = Flask(__name__)

def train_preprocessing(json):
    ...
    #TODO: Mit

def predict_preprocessing(json):
    ...

    #TODO: Mit

@app.route('/train/linear_regression', methods=['POST'])
def trainLinearRegression():
    ...
    #TODO: Ni Tran
    
@app.route('/train/naive_bayes', methods=['POST'])
def trainNaiveBayes():
    ...

@app.route('/train/logistics_regression', methods=['POST'])
def trainLogisticsRegression():
    ...

@app.route('/predict', methods=['POST'])
def predict():
    #TODO: Ni Tran

    ...    

if __name__ == '__main__':
    app.run(debug=True, port=5000)

