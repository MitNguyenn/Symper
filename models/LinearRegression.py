import os
import csv
from typing import Dict, List, Tuple

import uuid
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train(
    data: pd.DataFrame,
    target_columns: List[str],
    parameters: Dict[str, float | bool | str | List[float] | None]
) -> Tuple[str, Dict[str, float]]:
    """
    **Train a Linear Regression model and evaluate its performance.**

    Description:
        This function trains a Linear Regression model using the provided dataset and hyperparameters. It then 
        evaluates the model's performance and returns the model's ID along with key metrics.

    Parameters:
        data (pd.DataFrame): 
            A DataFrame containing features with numeric values (floats or integers).

        target_columns (List[str]): 
            A list of strings representing the names of the target columns to predict.

        parameters (Dict[str, float | bool | str | List[float] | None]): 
            A dictionary of hyperparameters for model training, with the following possible keys:

            - `test_size` (float): 
                The proportion of the dataset to reserve for validation. Should be between 0 and 1.
            - `fit_intercept` (bool, optional, default=True): 
                Whether to include an intercept in the model.
            - `positive` (bool, optional, default=True): 
                Whether to constrain the model's coefficients to be positive.
            - `ID_columns` (List[str]):
                An array of strings representing the index columns. Example: ['column1'] 

    Returns:
        `model_id` (str): 
            An identifier for the trained model that can be used for making predictions.
        `evaluation` (Dict[str, float]): 
            A dictionary containing performance metrics of the model:
            - `MSE` (float): 
                The Mean Squared Error of the model on the validation set. A lower value indicates a better model.
    """
    ID_column = parameters["ID_columns"]
    X = data.drop(columns=target_columns + ID_column)
    y = data[target_columns].copy()
    test_size = parameters["test_size"]

    if not test_size:
        raise ValueError("Missing test size")

    model = LinearRegression(fit_intercept = parameters.get('fit_intercept', True), positive = parameters.get('positive', True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)

    mse = mean_squared_error(model.predict(X_test), y_test)

    evaluation = {}
    evaluation['MSE'] = mse

    model_id = str(uuid.uuid4())

    new_row = [model_id, ",".join(ID_column), ",".join(target_columns)]

    with open("models/models.csv", "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(new_row)


    if not os.path.exists("save"):
        os.makedirs("save")
    joblib.dump(model, f"save/{model_id}.pkl")
    
    return model_id, evaluation


