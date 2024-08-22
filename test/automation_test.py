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
    df = pd.read_csv("test/"+filepath)
    data = df.values.tolist()
    data.insert(0, df.columns.tolist())
    return data

def generate_random_parameters(model_type, include_missing_params=False):
    parameters = {}
    if not include_missing_params or random.choice([True, False]):
        parameters["test_size"] = random.choice([(random.randint(-100, 100)/100)]+[random.choice(["0", True])])

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

def get_random_data():
    csv_string = random.choice(os.listdir("sample_data"))
    # csv_string = "LR_Student_Performance.csv"
    filepath = os.path.join(os.path.dirname(__file__), f'sample_data/{csv_string}')
    return filepath

def get_columns(filepath):
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        columns = next(reader)
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
    target = random.sample(columns, k=random.randint(0, len(columns)))
    id_columns = random.sample(columns, k=random.randint(0, len(columns)))

    parameters["ID_columns"] = id_columns

    input = {
        "data": data,
        "targets": target,
        "parameters": parameters
    }
    # print(data)
    # input["target"] = ["Performance Index"]
    # input["parameters"] = {}
    # input["parameters"]["ID_columns"] = ["Hours Studied"]
    # input["parameters"]["test_size"] = 0.2

    # json_file = json.dumps(input, indent=4)
    return filepath, input

def generate_random_predict_json():
    filepath = get_random_data()
    data = read_csv_and_convert(filepath)
    model_id = get_random_model_id()
    input = {
        "data" : data,
        "model_id" : model_id
    }

    return filepath, input

def test_train_linear_regression(client, filepath, json_file):
    try:
        print("Testing Linear regression")
        print("------------------------------------------------")
        print("Input: ")
        # inputfile = json.loads(json_file)
        # print(type(filepath))
        print("File: ", filepath.split('\\')[-1], end="\n\n")
        print(f"Target: {json_file['targets']}", end="\n\n")
        print(f"Parameters: ", end="\n")
        for i in json_file["parameters"]:
            print("   ", i, ": ", json_file["parameters"][i])
        print()
        print("--------------------------------------")
        print("Results")

        # filepath = os.path.join(os.path.dirname(__file__), 'sample_data/LR_Student_Performance.csv')
        # data = read_csv_and_convert(filepath)

        response = client.post('/train/linear_regression', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        print("Linear Regression Test Response: ", response_data)

        assert response.status_code == 200
        assert response_data['status'] == 'OK'
        assert 'model_id' in response_data
        assert 'evaluation' in response_data
    except AssertionError as e:
        print(f"Linear Regression Test Failed: {e}")
    # except Exception as e:
    #     print(f"An error occurred in Linear Regression Test: {e}")


    print()

def test_train_naive_bayes(client, filepath, json_file):
    try:
        print("Testing Naive Bayes")
        print("------------------------------------------------")
        print("Input: ")
        # inputfile = json.loads(json_file)
        # print(type(filepath))
        print("File: ", filepath.split('\\')[-1], end="\n\n")
        print(f"Target: {json_file['targets']}", end="\n\n")
        print(f"Parameters: ", end="\n")
        for i in json_file["parameters"]:
            print("   ", i, ": ", json_file["parameters"][i])
        print()
        print("--------------------------------------")
        print("Results")


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

def test_train_logistics_regression(client, filepath, json_file):
    try:
        print("Testing Logistic regression")
        print("------------------------------------------------")

        print("Input: ")
        # inputfile = json.loads(json_file)
        # print(type(filepath))
        print("File: ", filepath.split('\\')[-1], end="\n\n")
        print(f"Target: {json_file['targets']}", end="\n\n")
        print(f"Parameters: ", end="\n")
        for i in json_file["parameters"]:
            print("   ", i, ": ", json_file["parameters"][i])
        print()
        print("--------------------------------------")
        print("Results")

        
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

def test_predict(client, filepath, json_file):
    try:

        print("Testing Predict API")
        print("------------------------------------------------")

        print("Input: ")
        print()
        # inputfile = json.loads(json_file)
        # print(type(filepath))
        # print("File: ", filepath.split('\\')[-1], end="\n\n")
        print(f"Data Preview: ", end="\n")
        for i in json_file["data"][:10]:
            print(i)
        print()

        print(f"Model ID: {json_file['model_id']}", end="\n\n")
        print("--------------------------------------")
        print("Results")


        response = client.post('/predict', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        print("Prediction API Test Response:", response_data)
        if response_data["code"] == 200:
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
    type_testing = int(input("Type of testing: (0 for manually, 1 for automatically): "))
    if type_testing == 0:
        num_iter = 1
    else:
        num_iter = int(input("Number of iteration through testing: "))

    for _ in range(num_iter):
        print(f"TESTING NUMBER {_+1}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

        if type_testing == 1:
            lnr = generate_random_training_json(model_type="linear_regression")
            nb = generate_random_training_json(model_type="naive_bayes")
            lgr = generate_random_training_json(model_type="logistic_regression")
        
    

            test_train_linear_regression(client, lnr[0], lnr[1])
            print("================================================================================")
            test_train_naive_bayes(client, nb[0], nb[1])
            print("================================================================================")
            test_train_logistics_regression(client, lgr[0], lgr[1])
            print("================================================================================")

            predict = generate_random_predict_json()
            test_predict(client, predict[0], predict[1])

            print("================================================================================")
        
        else:
            model = input("Model: ")
            filepath = "sample_data/" + input("Data file name: ")+".csv"
            data = read_csv_and_convert(filepath)
            targets = []
            while True:
                target = input("Add target (Press enter to skip): ")
                if target:
                    targets.append(target)
                else:
                    break
            
            parameters = {}
            while True:
                name = input("Add parameter name (Press Enter to skip): ")
                if name:
                    if name == "ID_columns":
                        id_columns = []
                        while True:
                            id = input("Add data (Press Enter to skip): ")
                            if id:
                                id_columns.append(id)
                            else:
                                break
                        parameters[name] = id_columns
                    else:
                        dtype = input("Input data type: ")
                        value = input("Input value: ")
                        try:
                            if dtype in ["int"]:
                                parameters[name] = int(value)
                            elif dtype in ["float"]:
                                parameters[name] = float(value)
                            elif dtype in ["bool"]:
                                parameters[name] = bool(value)     
                        except TypeError:
                            print("Type Error, automatic convert to string")
                            parameters[name] = value               
                else:
                    break

            final_data = {
                "data" : data,
                "targets" : targets,
                "parameters" : parameters
            }

            if model == "linear_regression":
                test_train_linear_regression(client, filepath, final_data)
            elif model == "logistic_regression":
                test_train_logistics_regression(client, filepath, final_data)
            elif model == "naive_bayes":
                test_train_naive_bayes(client, filepath, final_data)




if __name__ == '__main__':
    test()
