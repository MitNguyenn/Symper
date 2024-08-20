import os
from typing import List, Dict, Tuple, Optional, Union


import uuid
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

def train(
    data: pd.DataFrame,
    target_columns: List[str],
    parameters: Dict[str, float | bool | str | None]
) -> Tuple[str, Dict[str, float]]:
    """
    **Train a Logistic Regression model and evaluate its performance.**

    Description:
        This function takes a DataFrame of features and target names, along with a set of parameters,
        to train a model and evaluate its performance. The trained model can then be used to make predictions on new data.

    Parameters:
        data (pd.DataFrame): 
            A DataFrame consisting of floats or integers, representing the features of the dataset.

        target_columns (List[str]): 
            A list of strings containing the names of the target columns to predict.

        parameters (Dict[str, float | bool | str | None]):
            A dictionary containing hyperparameters for model training. Keys include:
            
            - `test_size` (float, between 0 and 1):
                The proportion of the dataset to be used for validation. Defaults to 0.2.
            - `penalty` (str, optional, default="l2"):
                The type of regularization penalty to apply. Choices are "l1", "l2", "elasticnet", or "None".
            - `tol` (float, optional, default=1e-4):
                The tolerance for the stopping criteria of the optimization algorithm.
            - `C` (float, optional, default=1.0):
                The inverse of regularization strength. Must be positive. Smaller values indicate stronger regularization.
            - `fit_intercept` (bool, optional, default=True):
                Whether to include an intercept in the model.

    Returns:
        `model` (str): 
            The ID of the trained model that can be used for making predictions.

        `evaluation` (Dict[str, float]):
            A dictionary containing performance metrics of the model:
            
            + `accuracy` (float):
                The accuracy of the model on the validation set.
            + `precision` (float):
                The precision of the model on the validation set.
    """

    X = data.drop(columns=target_columns)
    y = data[target_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'])

    model = LogisticRegression(
        penalty=parameters.get('penalty', 'l2'),
        tol=parameters.get('tol', 1e-4),
        C=parameters.get('C', 1.0),
        fit_intercept=parameters.get('fit_intercept', True)
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    evaluation = {
        'accuracy': accuracy,
        'precision': precision
    }
    
    model_id = str(uuid.uuid4())
    model_id = f"{y.columns.to_list()}`~{model_id}"

    if not os.path.exists("save"):
        os.makedirs("save")
    joblib.dump(model, f"save/{model_id}.pkl")
    
    return model_id, evaluation

