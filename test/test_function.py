import sys
import os

os.chdir("..")
module_path = os.path.abspath(os.path.join('', ''))
sys.path.append(module_path)

import pandas as pd

from models.LinearRegression import train as train_linear_regression
from models.LogisticRegression import train as train_logistic_regression
from models.NaiveBayes import train as train_naive_bayes
from models.predict import predict
from main import train_preprocessing, predict_preprocessing

data = pd.DataFrame({
    "Feature1": [1.2, 2.3, 3.4, 4.5, 5.6],
    "Feature2": [3.4, 4.5, 5.6, 6.7, 7.8],
    "Feature3": [5.6, 6.7, 7.8, 8.9, 9.0],
    "Feature4": [1, 3, 2, 2, 1]
})


def test_linear_regression():
    print("TEST LINEAR REGRESSION\n")
    try:
        model_id, evaluation = train_linear_regression(data,
                                ["Feature3", "Feature4"], 
                                {
                                    "test_size" : 0.4,
                                    "fit_intercept" : True,
                                    "positive" : False
                                })
    except ValueError as e:
        if "could not convert string to float:" in str(e):
            print("Value Error: Invalid data type in data, data should only contain float/int")
        else:
            print(f"Value Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")

def test_logistic_regression():
    print("TEST LOGISTIC REGRESSION\n")
    try:
        model_id, evaluation = train_logistic_regression(data,
                                ["Feature4"], 
                                {
                                    "test_size" : 0.4,
                                    "penalty" : "l2",
                                    "tol" : 1e-4,
                                    "C" : 0.1,
                                    "fit_intercept" : True
                                })
    except ValueError as e:
        if "could not convert string to float:" in str(e):
            print("Value Error: Invalid data type in data, data should only contain float/int")
        else:
            print(f"Value Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")

def test_naive_bayes():
    print("TEST NAIVE BAYES\n")
    try:
        model_id, evaluation = train_naive_bayes(data,
                                ["Feature4"], 
                                {
                                    "test_size" : 0.4,
                                    "type" : "bernoulli",
                                    "alpha" : 1e-4
                                })
    except ValueError as e:
        if "could not convert string to float:" in str(e):
            print("Value Error: Invalid data type in data, data should only contain float/int")
        else:
            print(f"Value Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")

class Request:
    def __init__(self, request):
        self.request = request

    def get_json(self):
        return self.request

def test_train_preprocessing():
    print("TEST TRAIN PROCESSING\n")
    data = [["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"],
             [1.2, 2.3, 3.4, 4.5, 5.6],
             [3.4, 4.5, 5.6, 6.7, 7.8],
             [5.6, 6.7, 7.8, 8.9, 9.0],
             [7.8, 8.9, 9.0, 1.2, 2.3]
            ]

    request = Request(
        {"data" : data,
         "target" : ["Feature4"],
         "parameters" : {
             "none" : None
         }
         }
    )
    try:
        train_preprocessing(request)
    except ValueError as e:
        print(f"Value Error {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except Exception as e:
        print("An unexpected error occurred: ", e)

def test_predict_preprocessing():
    print("TEST PREDICT PROCESSING\n")
    data = [["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"],
             [1.2, 2.3, 3.4, 4.5, 5.6],
             [3.4, 4.5, 5.6, 6.7, 7.8],
             [5.6, 6.7, 7.8, 8.9, 9.0],
             [7.8, 8.9, 9.0, 1.2, 2.3]
            ]
    
    request = Request(
        {
            "data" : data,
            "model_id" : "lkfs-0fjoindlkfjn-jld.pkl"
        }
    )

    try:
        predict_preprocessing(request)
    except ValueError as e:
        print(f"Value Error {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except Exception as e:
        print("An unexpected error occurred: ", e)

    print("TEST PREDICT\n")
    data = pd.DataFrame({
        "Feature1": [1.2, 2.3, 3.4, 4.5, 5.6],
        "Feature2": [3.4, 4.5, 5.6, 6.7, 7.8],
    })

    model_id = "['Feature3', 'Feature4']`~c0d86d0a-1fb5-469c-854d-8d9e91915719"
    try:
        pred = predict(data, model_id)
    except FileNotFoundError:
        print("BRUH")
    except ValueError as e:
        print(f"Value Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

test_linear_regression()
test_train_preprocessing()
test_predict_preprocessing()
test_logistic_regression()
test_naive_bayes()
