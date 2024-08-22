import json
import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app
os.chdir("..")

def generate_random_parameters(model_type, include_missing_params=False):
    parameters = {}
    if not include_missing_params or random.choice([True, False]):
        parameters["test_size"] = random.choice([(random.randint(-100, 100)/100)]+["0", True])

    if model_type == "logistic_regression":        
        if not include_missing_params or random.choice([True, False]):
            parameters["penalty"] = random.choice(["l1", "l2", "elasticnet", "none"]+[1, 0, True, False])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["tol"] = random.uniform(-1e-2, 1e-2)
        
        if not include_missing_params or random.choice([True, False]):
            parameters["C"] = random.uniform(-10.00, 10.00)
        
        if not include_missing_params or random.choice([True, False]):
            parameters["fit_intercept"] = random.choice([True, False]+[1, 0, 10, "1", "0"])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["max_iter"] = random.choice([random.randint(100, 1000)] + ["0", True, str(random.randint(100, 1000))])
    
    elif model_type == "naive_bayes":        
        if not include_missing_params or random.choice([True, False]):
            parameters["model_type"] = random.choice(["gaussian", "multinomial", "bernoulli"] + ["complement", "categorical"] + ["none", 1, 0, -1, True, False])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["alpha"] = random.uniform(-10.00, 10.00)
    
    elif model_type == "linear_regression":

        if not include_missing_params or random.choice([True, False]):
            parameters["fit_intercept"] = random.choice([True, False] + [1, 0, "True", "False", 10])
        
        if not include_missing_params or random.choice([True, False]):
            parameters["positive"] = random.choice([True, False] + [1, 0, "True", "False", 10])
    
    return parameters

def generate_random_json(data, target, model_type, include_missing_params=False):
    parameters = generate_random_parameters(model_type=model_type, include_missing_params=include_missing_params)
    
    json_file = {
        "data": data,
        "target": target,
        "parameters": parameters
    }
    
    return json_file

def read_csv_and_convert(filepath: str):
    df = pd.read_csv(filepath)
    data = df.values.tolist()
    data.insert(0, df.columns.tolist())
    return data

def test_train_linear_regression(client, json_file):
    try:
        print("Testing Linear regression")
        print("------------------------------------------------")

        # filepath = os.path.join(os.path.dirname(__file__), 'sample_data/LR_Student_Performance.csv')
        # data = read_csv_and_convert(filepath)

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


    print()

def test_train_naive_bayes(client, json_file):
    try:
        print("Testing Naive Bayes")
        print("------------------------------------------------")

        # filepath = os.path.join(os.path.dirname(__file__), 'sample_data/suv_data.csv')
        # data = read_csv_and_convert(filepath)
        # json_file = {
        #     "data": data,
        #     "target": ["Purchased"],
        #     "parameters": {
        #         "test_size": 0.2,
        #         "model_type": "bernoulli",
        #         "alpha": 1e-9,
        #         "ID_columns": ["User ID"]
        #     }
        # }
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

    print()

def test_train_logistics_regression(client, json_file):
    try:
        print("Testing Logistic regression")
        print("------------------------------------------------")
        
        # filepath = os.path.join(os.path.dirname(__file__), 'sample_data/suv_data.csv')
        # data = read_csv_and_convert(filepath)
        # json_file = {
        #     "data": data,
        #     "target": ["Purchased"],
        #     "parameters": {
        #         "test_size": 0.2,
        #         "ID_columns": ["User ID"]
        #     }
        # }
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


    print()

def test_predict(client, json_file):
    try:
        # csv_file = "suv_data"
        # filepath = os.path.join(os.path.dirname(__file__), f'sample_data/{csv_file}.csv')
        # data = read_csv_and_convert(filepath)

        # data = [row[:-1] for row in data]
        # json_file = {
        #     "data": data, 
        #     "model_id": "badc4db5-ea6b-4871-9532-3d0bf677807b",
        # }
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



def test():
    app.config['TESTING'] = True
    client = app.test_client()

    # ...

    # test_train_linear_regression(client, json_file)
    # test_train_naive_bayes(client, json_file)
    # test_train_logistics_regression(client, json_file)
    # test_predict(client, json_file)

if __name__ == '__main__':
    test()
