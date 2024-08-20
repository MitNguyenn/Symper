import os

import uuid
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train(data, target_columns, params):
    """
        Summary of the function:
    
        Description:
        -----------------------
        This function returns a model that users can utilize to predict other data.
    
        Parameters:
        ------------------------
        data: pd.DataFrame
            A DataFrame consisting of floats/integers.
        target_columns: array
            An array of strings containing the target names.
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data.
            fit_intercept: bool (default: True)
                Whether to include an intercept in the model.
            positive: bool (default: True)
                Whether to constrain the model's coefficients to be positive.
            
        Returns:
        -------------------------
        model: string
            The ID of a model that can predict other unknown values.
        evaluation: dictionary
            MSE: float
                Mean Squared Error loss of the model. The smaller, the better.
    """

    X = data.drop(columns=target_columns)
    y = data[target_columns].copy()
    test_size = params["test_size"]

    model = LinearRegression(fit_intercept = params.get('fit_intercept', True), positive = params.get('positive', True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)

    mse = mean_squared_error(model.predict(X_test), y_test)

    evaluation = {}
    evaluation['MSE'] = mse

    model_id = str(uuid.uuid4())
    model_id = f"{y.columns.to_list()}`~{model_id}"

    if not os.path.exists("save"):
        os.makedirs("save")
    joblib.dump(model, f"save/{model_id}.pkl")
    
    return model_id, evaluation


