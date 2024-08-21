import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from models.NaiveBayes import train as train_naive_bayes
from models.LinearRegression import train as train_linear_regression
from models.LogisticRegression import train as train_logistic_regression
from models.predict import predict as pred

app = Flask(__name__)

def train_preprocessing(request):
    """
        Summary of the function.

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [column1, column2, column3, column4, column5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        targets: array
            An array of strings representing the target columns. Example: [column6, column7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of data used for validation.
            etc.: Additional parameters required by the model.

        Returns:
        ---------------------------------------
        data: pd.DataFrame
            The converted data in the form of a Pandas DataFrame.
        targets: np.array
            The converted target columns as a NumPy array.
        parameters: dictionary
            A dictionary containing the parameters used for the model.
    """

    input_data = request.get_json()
    try:
        data = input_data['data']
        target = input_data['target']
        parameters = input_data['parameters']
    except KeyError:
        raise KeyError("Request is missing parameters")

    df = pd.DataFrame(data[1:], columns=data[0])
    df.reset_index(drop=True, inplace=True)
    return df, target, parameters

def predict_preprocessing(request):
    """
        Summary of the function.

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [column1, column2, column3, column4, column5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        model_id: string
            The ID of the model.

        - ID_columns (List[str]):
            An array of strings representing the index columns. Example: ['column1']            


        Returns:
        ------------------------------------------
        data: pd.DataFrame
            The data converted into a Pandas DataFrame.
        model_id: string
            The ID of the model.
        ID: List[str]
            ID columns
    """

    input_data = request.get_json()
    try:
        data = input_data['data']
        model_id = input_data['model_id']
        ID = input_data['ID_columns']
    except KeyError:
        raise KeyError("Request is missing parameters")

    df = pd.DataFrame(data[1:], columns=data[0])
    df.reset_index(drop=True, inplace=True)
    return df, model_id, ID

@app.route('/train/linear_regression', methods=['POST'])
def trainLinearRegression():
    """
    **Trains a Linear Regression model and evaluates its performance.**

    Description:
        This function trains a Linear Regression model using the provided data, targets, and hyperparameters. 
        It then evaluates the model's performance and returns the model's ID along with key metrics.

    ## Parameters (request):
        `data` (List[List[Union[float, int, str]]]): 
            In the form of:
            [
                [column1, column2, column3, column4, column5, ...],\n
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...],\n
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],\n
                ...
            ]

        `targets` (List[str]): 
            An array of strings representing the target columns. Example: ['column4', 'column5']
        `parameters` (dictionary): 
            A dictionary of hyperparameters for model training, with the following possible keys:

            - ID_columns (List[str]):
                An array of strings representing the index columns. Example: ['column1']            
            - `test_size` (float): 
                The percentage of data to be used for validation. Must be between 0 and 1.
            - `fit_intercept` (bool, optional, default=True): 
                Whether to include an intercept (bias) in the model.
            - `positive` (bool, optional, default=True): 
                Whether to constrain the model's coefficients to be positive.

    ## Returns (JSON):
        model_id (string): 
            The ID of the model that can predict other unknown values.

        evaluation (dictionary): 
            A dictionary containing performance metrics of the model:

            - `MSE` (float): 
                The Mean Squared Error loss of the model. A lower value indicates a better model.
    """

    
    error = False
    try:
        data, targets, parameters = train_preprocessing(request)
    except ValueError as e:
        message = f"Value Error {e}"
        error = True
    except KeyError as e:
        message = f"Key Error: {e}"
        error = True
    except Exception as e:
        message =  f"An unexpected error occurred: {e}"
        error = True

    if not error:
        try:
            model_id, evaluation = train_linear_regression(data, targets, parameters)
        except ValueError as e:
            if "could not convert string to float:" in str(e):
                message =  "Value Error: Invalid data type in data, data should only contain float/int"
            else:
                message =  f"Value Error: {e}"
            error = True
        except KeyError as e:
            message =  f"Key Error: {e}"
            error = True
        except Exception as e:
            message =  f"An unexpected error occurred: {e}"
            error = True

    if error:
        return jsonify({
                    "status" : "error",
                    "message" : message,
                    "code" : 400
                }), 400

    return jsonify({
        "status" : "OK",
        "message" : "Data retrieved successfully",
        "code" : 200,
        "model_id": model_id,
        "evaluation": {
            "MSE": evaluation["MSE"]
            }
        })

@app.route('/train/logistics_regression', methods=['POST'])
def trainLogisticsRegression():
    """
    **Train a Logistic Regression model and evaluate its performance.**
    
    ## Parameters (request):
        `data` : List[List[Union[float, int, str]]]
            Input data structured as:
            [
                [column1, column2, column3, column4, column5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]

        `targets` : List[str]
            List of target column names. Example: [column4, column5]

        `parameters` : dict
            Dictionary of optional parameters for model training:
            - `test_size` (float): 
                Proportion of the dataset to include in the validation split. Must be between 0 and 1.
            - ID_columns (List[str]):
                An array of strings representing the index columns. Example: ['column1']            
            - `penalty` (str, optional, default="l2"): 
                Type of regularization to apply. Options are 'l1', 'l2', 'elasticnet', or 'None'.
            - `tol` (float, optional, default=1e-4): 
                Tolerance for stopping criteria. The algorithm stops when the loss function changes by less than this value.
            - `C` (float, optional, default=1.0): 
                Inverse of regularization strength. Must be positive. Smaller values specify stronger regularization.
            - `fit_intercept` (bool, optional, default=True): 
                Whether to include an intercept (bias) in the model.

    ## Returns (JSON):
        `model` : str
            Identifier for the trained model, which can be used to make predictions.

        `evaluation` : dict
            Dictionary with performance metrics of the model:

            - `accuracy` (float): 
                Accuracy of the model on the validation set.
            - `precision` (float):
                Precision of the model on the validation set.
    """
    error = False
    try:
        data, targets, parameters = train_preprocessing(request)
    except ValueError as e:
        message = f"Value Error {e}"
        error = True
    except KeyError as e:
        message = f"Key Error: {e}"
        error = True
    except Exception as e:
        message =  f"An unexpected error occurred: {e}"
        error = True

    if not error:
        try:
            model_id, evaluation = train_logistic_regression(data, targets, parameters)
        except ValueError as e:
            if "could not convert string to float:" in str(e):
                message =  "Value Error: Invalid data type in data, data should only contain float/int"
                error = True
            else:
                message = f"Value Error: {e}"
                error = True
        except KeyError as e:
            error = True
            message = f"Key Error: {e}"
        except Exception as e:
            message =  f"An unexpected error occurred: {e}"
            error = True

    
    if error:
        return jsonify({
                    "status" : "error",
                    "message" : message,
                    "code" : 400
                }), 400

    return jsonify({
        "status" : "OK",
        "message" : "Data retrieved successfully",
        "code" : 200,
        "model_id": model_id,
        "evaluation": {
            "accuracy": evaluation["accuracy"],
            "precision" : evaluation["precision"]
            }
        })

@app.route('/train/naive_bayes', methods=['POST'])
def trainNaiveBayes():
    """
    **Train a Naive Bayes model and evaluate its performance.**

    Description:
        This function trains a Naive Bayes model using the provided dataset and hyperparameters. 
        It then evaluates the model's performance and returns the model's ID along with key metrics.

    ## Parameters (request):
        `data` (List[List[Union[float, int, str]]]): 
            A list of lists where each inner list represents a row of features with numeric values (floats or integers).
            Example:
            [
                [column1, column2, column3, column4, column5, ...], \n
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], \n
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...], \n
                ...
            ]
            

        `targets` (List[str]): 
            A list of strings representing the target columns. Example: ['column4', 'column5']

        `parameters` (Dict[str, Union[float, str, List[float], None]]): 
            A dictionary of hyperparameters for model training, with the following possible keys:

            - `test_size` (float): 
                The percentage of data to be used for validation. Must be between 0 and 1.
            - ID_columns (List[str]):
                An array of strings representing the index columns. Example: ['column1']
            - `model_type` (str): 
                The type of Naive Bayes model to use. Options include "gaussian", "multinomial", or "bernoulli".
            - `priors` (List[float], optional, default=None): 
                The prior probabilities of the classes. If not provided, defaults to None, and priors will be estimated from the data.
            - `alpha` (float, optional, default=1e-9): 
                A smoothing parameter to ensure stability in calculations.

    ## Returns (JSON):
        A json file containing:
        - `model` (str): 
            The ID of the model that can predict other unknown values.

        - `evaluation` (Dict[str, float]): 
            A dictionary with performance metrics of the model:

            - `accuracy` (float): 
                The accuracy of the model on the validation set.
            - `precision` (float): 
                The precision of the model on the validation set.
    """
    
    error = False
    try:
        data, targets, parameters = train_preprocessing(request)
    except ValueError as e:
        message = f"Value Error {e}"
        error = True
    except KeyError as e:
        message = f"Key Error: {e}"
        error = True
    except Exception as e:
        message =  f"An unexpected error occurred: {e}"
        error = True

    if not error:
        try:
            model_id, evaluation = train_naive_bayes(data, targets, parameters)
        except ValueError as e:
            if "could not convert string to float:" in str(e):
                message = "Value Error: Invalid data type in data, data should only contain float/int"
                error = True
            else:
                message = f"Value Error: {e}"
                error = True
        except KeyError as e:
            message = f"Key Error: {e}"
            error = True
        except Exception as e:
            message =  f"An unexpected error occurred: {e}"
            error = True


    if error:
        return jsonify({
                    "status" : "error",
                    "message" : message,
                    "code" : 400
                }), 400

    return jsonify({
        "status" : "OK",
        "message" : "Data retrieved successfully",
        "code" : 200,
        "model_id": model_id,
        "evaluation": {
            "accuracy": evaluation["accuracy"],
            "precision" : evaluation["precision"]
            }
        })

@app.route('/predict', methods=['POST'])
def predict():

    """
    **Predicts outcomes using a trained model.**

    Parameters (request):
    ---------------------
    `data` : List[List[Union[float, int]]]
        Input data structured as:
        [
            [column1, column2, column3, column4, column5, ...], 
            [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
            [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
            ...
        ]
    `model_id` : str
        Identifier of the trained model used for making predictions.

    - ID_columns (List[str]):
        An array of strings representing the index columns. Example: ['column1']            


    Returns (JSON):
    ----------------
    List[List[Union[float, int]]]
        Predicted values in the same format as the input data:
        [
            [column6, column7, ...], 
            [x6_0, x7_0, ...], 
            [x6_1, x7_1, ...],
            ...
        ]
    """

    error = False

    try:
        df, model_id, id_columns = predict_preprocessing(request)
    except ValueError as e:
        message = f"Value Error {e}"
        error = True
    except KeyError as e:
        message = f"Key Error: {e}"
        error = True
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        error = True

    if not error:
        try:
            df_prediction = pred(df, model_id, id_columns)
        except FileNotFoundError as e:
            message = f"File Not Found Error: {e}"
            error = True
        except ValueError as e:
            message = f"Value Error: {e}"
            error = True
        except KeyError as e:
            message = f"Key Error: {e}"
            error = True
        
    if error:
        return jsonify({
            "status" : "error",
            "message" : message,
            "code" : 400,
            }), 400


    return jsonify({
        "status" : "OK",
        "message" : "Data retrieved successfully",
        "code" : 200,
        "prediction": df_prediction
        }), 200
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)

