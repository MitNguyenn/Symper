from typing import List

import joblib
import pandas as pd


def predict(X: pd.DataFrame, model_id: str, ID_columns: List[str]) -> pd.DataFrame:
    """
    **Returns predictions based on the provided model and input data.**

    Description:
        This function generates predictions using the specified model and the provided input data.

    Parameters:
        X (pd.DataFrame): 
            A DataFrame containing the input data for prediction.
        model_id (str): 
            The identifier of the model to be used for making predictions.
        ID_columns (List[str]):
                An array of strings representing the index columns. Example: ['column1'] 

    Returns:
        y (pd.DataFrame): 
            A DataFrame containing the predictions made by the model 
    """

    try:
        model = joblib.load(f"save/{model_id}.pkl")
        predictions = X[ID_columns].values.tolist()
        X.drop(columns=ID_columns, inplace=True)

        y = model.predict(X)

        y = y.tolist()
        for row,pred in zip(range(len(predictions)), y):
            predictions[row].append(pred)
        predictions.insert(0, [ID_columns, "predictions"])

        return predictions
    except FileNotFoundError:
        raise FileNotFoundError("No such model ID")

