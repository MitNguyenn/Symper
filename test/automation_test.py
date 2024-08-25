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

def check_input(json_file, model):
    print(model, ": ", end="")
    data = json_file["data"]
    length = len(data[0])
    columns = data[0]

    for i in data[1:]:
        if len(i) != length:
            print("RETURN 400 data wrong length")
            return 400
        for j in i:
            if type(j) not in [float, int]:
                print("RETURN 400 data type wrong")
                return 400

    if model != "predict":
        targets = json_file["targets"]
        parameters = json_file["parameters"]

        if "ID_columns" in parameters:
            for ID_columns in parameters["ID_columns"]: 
                if ID_columns not in columns:
                    print("RETURN 400 ID col not in data")
                    return 400

        if not targets:
            print("RETURN 400 no targets")
            return 400
        elif len(set(targets+parameters["ID_columns"])) == len(columns):
            print("RETURN 400 must have at least 1 feature left")
            return 400
        else:
            for target in targets:
                if target not in columns:
                    print("RETURN 400 target not in data")
                    return 400

                
        if  "test_size" not in parameters:
            print("RETURN 400 no test_size")
            return 400
        else:
            # print(type(parameters["test_size"]), type(parameters["test_size"]) not in [float, int], parameters["test_size"])
            if (type(parameters["test_size"]) not in [float, int]) or (parameters["test_size"] >= 1) or (parameters["test_size"] <= 0):
                print("RETURN 400 invalid test_size")
                return 400
        
        for i in ["positive", "fit_intercept"]:
            if i in parameters:
                if type(parameters[i]) != bool:
                    print("RETURN 400 invalid positive/fit_intercept")
                    return 400
        
        for i in ["C", "tol", "alpha"]:
            if i in parameters:
                if type(parameters[i]) not in [float, int]:
                    print("RETURN 400 invalid C/tol/alpha")
                    return 400
        for i in ["C", "alpha", "tol"]:
            if i in parameters:
                if parameters[i] < 0:
                    print("RETURN 400 invalid C/alpha")
                    return 400
            
        if "penalty" in parameters:
            if parameters["penalty"] not in ['l1', 'l2', 'elasticnet','None']:
                print("RETURN 400 invalid penalty")
                return 400
            
        
        if model == "naive_bayes":
            if "model_type" not in parameters:
                print("RETURN 400 missing model_type")
                return 400
            elif parameters["model_type"] not in ["bernoulli", "gaussian", "multinomial"]:
                print("RETURN 400 invalid model_type")
                return 400
            
            if len(targets) != 1:
                print("RETURN 400 TOO much y")
                return 400

    else:
        model_id = json_file["model_id"]
        with open("models/models.csv", "r", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['model_id'] == model_id:
                    ID_columns = row["ID_columns"].split(",")
                    target_columns = row["target_columns"].split(",")

        # print(columns, ID_columns, target_columns)
        for i in [ID_columns, target_columns]:
            for col in i:
                if col not in columns:
                    # print("RETURN 400")
                    print("RETURN 400 Invalid/Wrong model")
                    return 400
                
        
    print("PASS")
    return 200
            
    
def read_csv_and_convert(filepath: str):
    # print(os.listdir(), os.getcwd())
    df = pd.read_csv(filepath)
    data = df.values.tolist()
    data.insert(0, df.columns.tolist())

    # print(data)
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
    filepath = f'sample_data/{csv_string}'
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

def test_train_linear_regression(client, filepath, json_file, _print=False):
    try:
        if _print:

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


        response = client.post('/train/linear_regression', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        if _print:
            print("Linear Regression Test Response: ", response_data)

        assert response.status_code == 200
        assert response_data['status'] == 'OK'
        assert 'model_id' in response_data
        assert 'evaluation' in response_data
        return 200
    except AssertionError as e:
        # print(f"Linear Regression Test Failed: {e}")
        pass
    except Exception as e:
        pass
        # print(f"An error occurred in Linear Regression Test: {e}")

    # print()
    return 400

def test_train_naive_bayes(client, filepath, json_file, _print=False):
    try:
        if _print:

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


        response = client.post('/train/naive_bayes', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        if _print:

            print("Naive Bayes Test Response:", response_data)

        assert response.status_code == 200
        assert response_data['status'] == 'OK'
        assert 'model_id' in response_data
        assert 'evaluation' in response_data
        return 200
    except AssertionError as e:
        # print(f"Naive Bayes Test Failed: {e}")
        pass
    except Exception as e:
        pass
        # print(f"An error occurred in Naive Bayes Test: {e}")
    return 400
    # print()

def test_train_logistics_regression(client, filepath, json_file, _print=False):
    try:
        if _print:
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

        
        response = client.post('/train/logistics_regression', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        if _print:
            print("Logistic Regression Test Response:", response_data)

        assert response.status_code == 200
        assert response_data['status'] == 'OK'
        assert 'model_id' in response_data
        assert 'evaluation' in response_data
        return 200
    except AssertionError as e:
        # print(f"Logistic Regression Test Failed: {e}")
        pass
    except Exception as e:
        # print(f"An error occurred in Logistic Regression Test: {e}")
        pass

    return 400

def test_predict(client, filepath, json_file, _print=False):
    try:
        if _print:
            print("Testing Predict API")
            print("------------------------------------------------")

            print("Input: ")
            print()
            # inputfile = json.loads(json_file)
            # print(type(filepath))
            print("File: ", filepath.split('\\')[-1], end="\n\n")
            print(f"Data Preview: ", end="\n")
            for i in json_file["data"][:10]:
                print(i)
            print()

            real_data = json_file["data"]

            print(f"Model ID: {json_file['model_id']}", end="\n\n")
            print("--------------------------------------")
            print("Results")


        response = client.post('/predict', data=json.dumps(json_file), content_type='application/json')
        response_data = response.json
        if response_data["code"] != 200 and _print:
            print("Prediction API Test Response:", response_data)
        # if response_data["code"] == 200:
        #     prediction = response_data["prediction"]
            # print("\nPrediction Preview")
            # for el in range(len(prediction[:10])):
            #     print(prediction[el])
            #     print(real_data[el])
        assert response.status_code == 200
        assert response_data['status'] == 'OK'
        assert 'prediction' in response_data
        return 200
    except AssertionError as e:
        # print(f"Prediction Test Failed: {e}")
        pass
    except Exception as e:
        # print(f"An error occurred in Prediction Test: {e}")
        pass
    return 400

def test():
    app.config['TESTING'] = True
    client = app.test_client()
    type_testing = int(input("Type of testing: (0 for manually, 1 for automatically): "))
    fail = []
    if type_testing == 0:
        num_iter = 1
    else:
        num_iter = int(input("Number of iteration through testing: "))

    correct = 0
    for _ in range(num_iter):
        print(f"TESTING NUMBER {_}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

        if type_testing == 1:
            lnr = generate_random_training_json(model_type="linear_regression")
            nb = generate_random_training_json(model_type="naive_bayes")
            lgr = generate_random_training_json(model_type="logistic_regression")
        
#----------------------------------------------------------------------------------------------------------

            code_lnr = test_train_linear_regression(client, lnr[0], lnr[1])
            code_check_lnr = check_input(lnr[1], model="linear_regression")
            if code_lnr == code_check_lnr:
                correct += 1 
            else:
                fail.append(f"Test {_} - Linear Regression")
                test_train_linear_regression(client, lnr[0], lnr[1], _print=True)
#----------------------------------------------------------------------------------------------------------

            code_nb = test_train_naive_bayes(client, nb[0], nb[1])
            code_check_nb = check_input(nb[1], model="naive_bayes")
            if code_nb == code_check_nb:
                correct += 1 
            else:
                fail.append(f"Test {_} - Naive Bayes")
                test_train_naive_bayes(client, nb[0], nb[1], _print=True)
                
#----------------------------------------------------------------------------------------------------------

            code_lgr = test_train_logistics_regression(client, lgr[0], lgr[1])
            code_check_lgr = check_input(lgr[1], model="logistic_regression")
            if code_lgr == code_check_lgr:
                correct += 1 
            else:
                fail.append(f"Test {_} - Logistic Regression")
                test_train_logistics_regression(client, lgr[0], lgr[1], _print=True)
#----------------------------------------------------------------------------------------------------------


            predict = generate_random_predict_json()
            code_predict = test_predict(client, predict[0], predict[1])
            code_check_predict = check_input(predict[1], model="predict")

            if code_predict == code_check_predict:
                correct += 1 
            else:
                fail.append(f"Test {_} - Predict")
                test_predict(client, predict[0], predict[1], _print=True)



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
                    dtype = input("Input data type: ")
                    if dtype == "list_string":
                        parameters[name] = []
                        while True:
                            value = input("Input value: ")
                            if not value:
                                break
                            else:
                                parameters[name].append(value)
                    elif dtype == "list_float":
                        parameters[name] = []
                        while True:
                            value = input("Input value: ")
                            if not value:
                                break
                            else:
                                parameters[name].append(float(value))
                    else:
                        value = input("Input value: ")
                        if dtype in ["int"]:
                            parameters[name] = int(value)
                        elif dtype in ["float"]:
                            parameters[name] = float(value)
                        elif dtype in ["bool"]:
                            parameters[name] = bool(value)     
                        elif dtype in ["string"]:
                            parameters[name] = str(value)
                else:
                    break

            final_data = {
                "data" : data,
                "targets" : targets,
                "parameters" : parameters
            }

            # print(final_data["parameters"])

            if model == "linear_regression":
                test_train_linear_regression(client, filepath, final_data)
            elif model == "logistic_regression":
                test_train_logistics_regression(client, filepath, final_data)
            elif model == "naive_bayes":
                test_train_naive_bayes(client, filepath, final_data)

            
            model_id = input("Model ID: ")
            filepath = "sample_data/" + input("Data file name: ")+".csv"
            data = read_csv_and_convert(filepath)

            final_data = {
                "model_id" : model_id,
                "data" : data
            }

            test_predict(client, filepath, final_data)
    
    if type_testing == 1:
        print(f"CHECKING: PASS {correct}/{num_iter*4}")
        if correct != num_iter*4:
            print("Fail at: \n")
            for i in fail:
                print(i)
if __name__ == '__main__':
    test()
