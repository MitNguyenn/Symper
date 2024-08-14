from flask import Flask, request, jsonify
import pandas as pd
from models.NaiveBayes import train as train_naive_bayes
from models.LinearRegression import train as train_linear_regression
from models.LogisticRegression import train as train_logistic_regression
from models.predict import predict as pred
import uuid

app = Flask(__name__)

def train_preprocessing(request):
    """
        Summary of function

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        targets: array
            An array of string consisting the target columns. [col6, col7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            etc. (Every parameter that the model need)

        Returns (json):
        ---------------------------------------
        data: pd.DataFrame
            converted data into form of pd.DataFrame
        targets: np.array
            Converted targets columns into np.array
                parameters: dictionary
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            etc. (Every parameter that the model need)
    """


def predict_preprocessing(request):
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        model_id: string
            The id of the model

        Returns (json):
        ------------------------------------------
        data: pd.DataFrame
            converted data into form of pd.DataFrame
        model_id: string
            The id of the model
    """


    #TODO: Mit

@app.route('/train/linear_regression', methods=['POST'])
def trainLinearRegression():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        targets: array
            An array of string consisting the target columns. [col6, col7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            fit_intercept=True: bool
                Weather to have bias in the model or not
            positive=True: bool
                Weather to set bias to always be positive or not

        Returns (json):
            model_id : string
                The id of a model that can predict other unknown value
            evaluation: dictionary
                MSE:
                    Mean Squared Error loss of the model. The smaller, the better

    """
    data, targets, parameters = train_preprocessing(request)

    model_id, evaluation = train_linear_regression(data, targets, parameters)

    return jsonify({
        "model_id": model_id,
        "evaluation": {
            "MSE": evaluation["MSE"]
            }
        })

    #TODO: Ni Tran
    
@app.route('/train/naive_bayes', methods=['POST'])
def trainNaiveBayes():
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        targets: array
            An array of string consisting the target columns. [col6, col7]
        parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            model_type: string (gaussian/multinomial/bernoulli)
                Type of Naive Bayes model used
            priors=None: array (Shape = target.shape)
                Prior probabilities of the classes
            alpha=1e-9: float
                A number create stability for calculating

        Returns (json):
            model : string
                The id of a model that can predict other unknown value
            evaluation: dictionary
                accuracy: float
                    The accuracy of the model
                precision: float
                    The preciion of model
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
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        targets: array
            An array of string consisting the target columns. [col6, col7]
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            penalty="l2": string ("l1", "l2", "elasticnet", "None")
                The normalization of the penalty
            tol=1e-4: float 
                Tolerance for stopping criteria
            C=1.0: float (must be positive)
                Inverse of regularization strength: The smaller the number, the stronger the regularization
            fit_intercept=True: bool
                Weather to have bias in the model or not 

        Returns (json):
            model : string
                The id of a model that can predict other unknown value
            evaluation: dictionary
                accuracy:
                    The accuracy of the model
                precision: 
                    The preciion of the model
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
def predict(request):
    """
        Summary of API

        Parameters (request):
        ------------------------------------------
        data: array of float/int
            In form of [
                        [col1, col2, col3, col4, col5, ...], 
                        [x1_0, x2_0, x3_0, x4_0, x5_0, ...], 
                        [x1_1, x2_1, x3_1, x4_1, x5_2, ...],
                        ....             
                        ]
        model_id: string
            The id of the model

        Returns (json):
        ------------------------------------------
        prediction: array of int/float
            In form of [
                        [col6, col7 ...], 
                        [x6_0, x7_0, ...], 
                        [x6_1, x7_1, ...],
                        ....             
                        ]


    """
    data, model_id = predict_preprocessing(request)

    columns = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=columns)
    
    df_prediction = pred(df, model_id)

    prediction = [df_prediction.columns.tolist()] + df_prediction.values.tolist()

    return jsonify({
        "prediction": prediction
        )}
    #TODO: Ni Tran
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)

