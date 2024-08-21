import csv
from typing import List

import joblib
import pandas as pd


def predict(X: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    **Returns predictions based on the provided model and input data.**

    Description:
        This function generates predictions using the specified model and the provided input data.

    Parameters:
        X (pd.DataFrame): 
            A DataFrame containing the input data for prediction.
        model_id (str): 
            The identifier of the model to be used for making predictions.

    Returns:
        y (pd.DataFrame): 
            A DataFrame containing the predictions made by the model 
    """

    try:
        with open("models/models.csv", "r", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['model_id'] == model_id:
                    ID_columns = row["ID_columns"].split(",")
                    target_columns = row["target_columns"].split(",")


        model = joblib.load(f"save/{model_id}.pkl")
        predictions = X[ID_columns].values.tolist()
        X.drop(columns=ID_columns, inplace=True)
        
        y = model.predict(X)

        y = y.tolist()
        multiple = False

        if type(y[0]) == list:
            multiple = True
        for row,pred in zip(range(len(predictions)), y):
            if not multiple:
                predictions[row].append(pred)
            else:
                predictions[row] += pred
        predictions.insert(0, [i for i in ID_columns] + [i for i in target_columns])

        return predictions
    except FileNotFoundError:
        raise FileNotFoundError("No such model ID")

