from sklearn.externals import joblib

def predict(X, model_id):
    """
        Summary of function
    
        Description:
        -----------------------
        Returns predictions based on the provided model and input data.
    
        Parameters:
        ------------------------
        X: pd.DataFrame
            A DataFrame containing the input data for prediction.
        model_id: string
            The identifier of the model to be used for making predictions.
    
        Returns:
        -------------------------
        y: pd.DataFrame
            A DataFrame containing the predictions made by the model.
    """

    model = joblib.load(f"save/{model_id}.pkl")
    try:
        y = model.predict(X)
        return y
    except Exception:
        return("Incorrect model or Invalid model")

    return 0
