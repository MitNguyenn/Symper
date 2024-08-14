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
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        targets: array
            An array of strings representing the target columns. Example: [col6, col7]
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

    input_data = np.array(request.get_json())
    data = input_data['data']
    target = input_data['target']
    parameters = input_data['parameters']

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
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        model_id: string
            The ID of the model.

        Returns:
        ------------------------------------------
        data: pd.DataFrame
            The data converted into a Pandas DataFrame.
        model_id: string
            The ID of the model.
    """

    input_data = np.array(request.get_json())
    data = input_data['data']
    model_id = input_data['model_id']

    df = pd.DataFrame(data[1:], columns=data[0])
    df.reset_index(drop=True, inplace=True)
    return df, model_id

@app.route('/train/linear_regression', methods=['POST'])
def trainLinearRegression():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        targets: array
            An array of strings representing the target columns. Example: [col6, col7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of data used for validation.
            fit_intercept: bool, optional (default=True)
                Whether to include an intercept (bias) in the model.
            positive: bool, optional (default=True)
                Whether to constrain the model's coefficients to be positive.

        Returns (JSON):
        ------------------------------------------
        model_id: string
            The ID of the model that can predict other unknown values.
        evaluation: dictionary
            MSE: float
                Mean Squared Error loss of the model. The smaller, the better.
    """

    data, targets, parameters = train_preprocessing(request)

    model_id, evaluation = train_linear_regression(data, targets, parameters)

    return jsonify({
        "model_id": model_id,
        "evaluation": {
            "MSE": evaluation["MSE"]
            }
        })

@app.route('/train/naive_bayes', methods=['POST'])
def trainNaiveBayes():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        targets: array
            An array of strings representing the target columns. Example: [col6, col7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of data used for validation.
            model_type: string (options: "gaussian", "multinomial", "bernoulli")
                The type of Naive Bayes model to be used.
            priors: array, optional (default=None) (shape = target.shape)
                Prior probabilities of the classes.
            alpha: float, optional (default=1e-9)
                Smoothing parameter to ensure stability in calculations.

        Returns (JSON):
        ------------------------------------------
        model: string
            The ID of the model that can predict other unknown values.
        evaluation: dictionary
            accuracy: float
                The accuracy of the model.
            precision: float
                The precision of the model.
    """
    
    data, targets, parameters = train_preprocessing(request)

    model_id, evaluation = train_naive_bayes(data, targets, parameters)

    return jsonify({
        "model_id": model_id,
        "evaluation": {
            "accuracy": evaluation["accuracy"],
            "precision" : evaluation["precision"]
            }
        })

@app.route('/train/logistics_regression', methods=['POST'])
def trainLogisticsRegression():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        targets: array
            An array of strings representing the target columns. Example: [col6, col7]
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of data used for validation.
            penalty: string, optional (default="l2") ("l1", "l2", "elasticnet", "None")
                The type of regularization penalty.
            tol: float, optional (default=1e-4)
                Tolerance for stopping criteria.
            C: float, optional (default=1.0, must be positive)
                Inverse of regularization strength. The smaller the number, the stronger the regularization.
            fit_intercept: bool, optional (default=True)
                Whether to include an intercept (bias) in the model.

        Returns (JSON):
        ------------------------------------------
        model: string
            The ID of the model that can predict other unknown values.
        evaluation: dictionary
            accuracy: float
                The accuracy of the model.
            precision: float
                The precision of the model.
    """

    data, targets, parameters = train_preprocessing(request)

    model_id, evaluation = train_logistic_regression(data, targets, parameters)

    return jsonify({
        "model_id": model_id,
        "evaluation": {
            "accuracy": evaluation["accuracy"],
            "precision" : evaluation["precision"]
            }
        })
    
@app.route('/predict', methods=['POST'])
def predict():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In the form of:
            [
                [col1, col2, col3, col4, col5, ...], 
                [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                [x1_1, x2_1, x3_1, x4_1, x5_1, ...],
                ...
            ]
        model_id: string
            The ID of the model.

        Returns (JSON):
        ------------------------------------------
        prediction: array of int/float
            In the form of:
            [
                [col6, col7, ...], 
                [x6_0, x7_0, ...], 
                [x6_1, x7_1, ...],
                ...
            ]
    """

    data, model_id = predict_preprocessing(request)

    columns = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=columns)
    df = df.reset_index(drop=True)
    
    df_prediction = pred(df, model_id)

    prediction = [df_prediction.columns.tolist()] + df_prediction.values.tolist()

    return jsonify({
        "prediction": prediction
        })
 
if __name__ == '__main__':

    app.run(debug=True, port=5000)

