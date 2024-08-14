from flask import Flask, request, jsonify
import pandas as pd
from models.NaiveBayes import train as train_naive_bayes
from models.LinearRegression import train as train_linear_regression
from models.LogisticRegression import train as train_logistic_regression
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
            fit_intercept=True: bool
                Weather to have bias in the model or not
            positive=True: bool
                Weather to set bias to always be positive or not

        Returns (json):
            data: pd.DataFrame
                converted data into form of pd.DataFrame
            targets: np.array
                Converted targets columns into np.array
                    parameters: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            fit_intercept=True: bool
                Weather to have bias in the model or not
            positive=True: bool
                Weather to set bias to always be positive or not            
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
    df, model_id = predict_preprocessing(request)

    targets = request.json.get('targets')
    parameters = request.json.get('parameters')

    if not targets or not parameters:
        return jsonify({"error": "Missing targets or parameters in the request"}), 400
    
    X = df.drop(columns=targets)
    y = df[targets]

    test_size = parameters.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = joblib.load(f"models/{model_id}.pkl")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return jsonify({
        "model": model_id,
        "evaluation": {
            "MSE": mse
        }

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
    #TODO: Ni Tran


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
    
    json_data = request.get_json()

    data = pd.DataFrame(json_data['data'])
    targets = json_data['targets']
    params = json_data['params']

    model_id, evaluation = train_logistic_regression(data, targets, params)

    return jsonify({
        "model": model_id,
        "evaluation": evaluation
    })


    #TODO: Ni Tran
    

@app.route('/predict', methods=['POST'])
def predict():
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
    #TODO: Ni Tran
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)

