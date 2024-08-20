

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
            A DataFrame containing the predictions made by the model.
    """

    try:
        model = joblib.load(f"save/{model_id}.pkl")
        columns = model_id.split("`~")[0][1:-1]
        columns = columns.replace("'", "").split(", ")
        y = model.predict(X)

        y = y.tolist()
        y.insert(0, columns)
        return y
    except FileNotFoundError:
        raise FileNotFoundError("No such model ID")

