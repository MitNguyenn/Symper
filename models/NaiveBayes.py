import os
from typing import List, Dict, Tuple
import csv

import uuid
import joblib
import pandas as pd

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score

def train(
    data: pd.DataFrame,
    target_columns: List[str],
    parameters: Dict[str, float | str | List[float] | None]
) -> Tuple[str, Dict[str, float]]:
    """
    **Train a Naive Bayes model and evaluate its performance.**

    Description:
        This function trains a Naive Bayes model using the provided dataset and hyperparameters. 
        It then evaluates the model's performance and returns the model's ID along with key metrics.

    Parameters:
        data (pd.DataFrame): 
            A DataFrame containing features with numeric values (floats or integers).

        target_columns (List[str]): 
            A list of strings representing the names of the target columns to predict.

        parameters (Dict[float | str | List[float] | None]): 
            A dictionary of hyperparameters for model training, with the following possible keys:
            
            - `test_size` (float): 
                The proportion of the dataset to reserve for validation. Must be between 0 and 1.
            - `model_type` (str): 
                The type of Naive Bayes model to use. Options include 'gaussian', 'multinomial', or 'bernoulli'.
            - `priors` (List[float], optional): 
                The prior probabilities of the classes. If not provided, defaults to None, and priors will be estimated from the data.
            - `alpha` (float, optional, default=1e-9): 
                A smoothing parameter to handle zero probabilities in calculations.

    Returns:
        `model_id` (str): 
            An identifier for the trained model that can be used for making predictions.

        `evaluation` (Dict[str, float]): 
            A dictionary containing performance metrics of the model:
            
            - `accuracy` (float): 
                The accuracy of the model on the validation set.
            - `precision` (float): 
                The precision of the model on the validation set.
            - `ID_columns` (List[str]):
                An array of strings representing the index columns. Example: ['column1'] 
    """
    
    test_size = parameters.get('test_size', 0.2)
    type = parameters.get('model_type')
    alpha = parameters.get('alpha', 1e-9)
    priors = parameters.get('priors', None)
    ID_column = parameters['ID_columns']


    X = data.drop(columns=target_columns + ID_column)
    y = data[target_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if type.lower() == "gaussian":
        model = GaussianNB(var_smoothing=alpha, priors=priors)
    elif type.lower() == "multinomial":
        for col in y.columns:
            if len(y[col].unique()) != 2:
                raise ValueError("Targets must only have 2 unique values") 
        model = MultinomialNB(alpha=alpha, class_prior=priors)
    elif type.lower() == "bernoulli":
        model = BernoulliNB(alpha=alpha, class_prior=priors)
    else:
        raise ValueError("Invalid Naive Bayes model type")


    model.fit(X_train, y_train)

    evaluation = {}

    evaluation["accuracy"] = accuracy_score(model.predict(X_test), y_test)
    evaluation["precision"] = precision_score(model.predict(X_test), y_test)

    model_id = str(uuid.uuid4())

    new_row = [model_id, ",".join(ID_column), ",".join(target_columns)]

    with open("models/models.csv", "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(new_row)

    if not os.path.exists("save"):
        os.makedirs("save")
    joblib.dump(model, f"save/{model_id}.pkl")
    

    return model_id, evaluation

