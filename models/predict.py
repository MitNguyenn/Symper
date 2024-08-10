from sklearn.externals import joblib

def predict(X, model_id):
    """
        Summary of function.

        Description:
        -----------------------
        Return the prediction

        Parameters (json file):
        ------------------------
        X: pd.DataFrame
            A DataFrame containing the X data needed to be predicted
        model_id
            The model_id to get the model

        Returns (json file):
        -------------------------
        y: pd.DataFrame
            The prediction of the model
    """

    model = joblib.load(f"save/{model_id}")
    try:
        y = model.predict(X)
        return y
    except Exception:
        print("Incorrect model or Invalid model")

    return 0