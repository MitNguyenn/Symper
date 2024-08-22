import json
import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import csv

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app
os.chdir("..")


def read_csv_and_convert(filepath: str):
    df = pd.read_csv(filepath)
    data = df.values.tolist()
    data.insert(0, df.columns.tolist())
    return data

def generate_random_parameters(model_type, include_missing_params=False):
    parameters = {}
    if model_type == "logistic_regression":
        if not include_missing_params or random.choice([True, False]):
            parameters["test_size"] = random.choice([0.1, 0.2, 0.3, 0.4])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["penalty"] = random.choice(["l1", "l2", "elasticnet", "none"])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["tol"] = random.uniform(1e-4, 1e-2)
        
        if not include_missing_params or random.choice([True, False]):
            parameters["C"] = random.uniform(0.1, 10.0)
        
        if not include_missing_params or random.choice([True, False]):
            parameters["fit_intercept"] = random.choice([True, False])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["max_iter"] = random.randint(100, 1000)
    
    elif model_type == "naive_bayes":
        if not include_missing_params or random.choice([True, False]):
            parameters["test_size"] = random.choice([0.1, 0.2, 0.3, 0.4])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["model_type"] = random.choice(["gaussian", "multinomial", "bernoulli"])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["alpha"] = random.uniform(1e-9, 1.0)
    
    elif model_type == "linear_regression":
        if not include_missing_params or random.choice([True, False]):
            parameters["test_size"] = random.choice([0.1, 0.2, 0.3, 0.4])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["fit_intercept"] = random.choice([True, False])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["positive"] = random.choice([True, False])
    
    if not include_missing_params or random.choice([True, False]):
        parameters["ID_columns"] = random.choice([[], ["User ID"], ["User ID", "Gender"]])
    
    return parameters

def get_random_data():
    csv_string = random.choice(["LR_Student_Performance", "suv_data", "winequality-red", "winequality-white", "error-causing"])
    filepath = os.path.join(os.path.dirname(__file__), f'../sample_data/{csv_string}.csv')
    return filepath

def get_columns(filepath):
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        columns = next(reader)  # Get the first row (column headers)
    return columns

def get_random_model_id():
    filepath = os.path.join(os.path.dirname(__file__), f'../models/models.csv')
    data = []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row["model_id"])
    model_id = random.choice(data)
    return model_id

def generate_random_training_json(model_type, include_missing_params=False):
    parameters = generate_random_parameters(model_type=model_type, include_missing_params=include_missing_params)
    filepath = get_random_data()
    data = read_csv_and_convert(filepath)
    columns = get_columns(filepath)
    target = random.choice(columns)
    input = {
        "data": data,
        "target": target,
        "parameters": parameters
    }
    json_file = json.dumps(input, indent=4)
    return json_file

def generate_random_predict_json(model_type, include_missing_params=False):
    filepath = get_random_data()
    data = read_csv_and_convert(filepath)
    model_id = get_random_model_id() 
    input = {
        "data": data,
        "model_id": model_id
    }
    json_file = json.dumps(input, indent=4)
    return json_file   

def test():
    app.config['TESTING'] = True
    client = app.test_client()

    def test_train_linear_regression():
        try:
            filepath = os.path.join(os.path.dirname(__file__), '../sample_data/LR_Student_Performance.csv')
            data = read_csv_and_convert(filepath)
            json_file = {
                "data": data,
                "target": ["Performance Index"],
                "parameters": {
                    "test_size": 0.2,
                    "fit_intercept": True,
                    "positive": True,
                    "ID_columns": []
                }
            }
            response = client.post('/train/linear_regression', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Linear Regression Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Linear Regression Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Linear Regression Test: {e}")

    def test_train_naive_bayes():
        try:
            filepath = os.path.join(os.path.dirname(__file__), '../sample_data/suv_data.csv')
            data = read_csv_and_convert(filepath)
            json_file = {
                "data": data,
                "target": ["Purchased"],
                "parameters": {
                    "test_size": 0.2,
                    "model_type": "bernoulli",
                    "alpha": 1e-9,
                    "ID_columns": ["User ID"]
                }
            }
            response = client.post('/train/naive_bayes', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Naive Bayes Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Naive Bayes Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Naive Bayes Test: {e}")

    def test_train_logistics_regression():
        try:
            filepath = os.path.join(os.path.dirname(__file__), '../sample_data/suv_data.csv')
            data = read_csv_and_convert(filepath)
            json_file = {
                "data": data,
                "target": ["Purchased"],
                "parameters": {
                    "test_size": 0.2,
                    "ID_columns": ["User ID"]
                }
            }
            response = client.post('/train/logistics_regression', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Logistic Regression Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Logistic Regression Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Logistic Regression Test: {e}")

    def test_predict():
        try:
            csv_file = "suv_data"
            filepath = os.path.join(os.path.dirname(__file__), f'../sample_data/{csv_file}.csv')
            data = read_csv_and_convert(filepath)

            data = [row[:-1] for row in data]
            json_file = {
                "data": data, 
                "model_id": "badc4db5-ea6b-4871-9532-3d0bf677807b",
            }
            response = client.post('/predict', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            prediction = response_data["prediction"]
            for el in prediction:
                print(el)
            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'prediction' in response_data
        except AssertionError as e:
            print(f"Prediction Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Prediction Test: {e}")

    test_train_linear_regression()
    test_train_naive_bayes()
    test_train_logistics_regression()
    test_predict()

if __name__ == '__main__':
    test()
