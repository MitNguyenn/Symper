import joblib

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
    except Exception:
        return("Incorrect model or Invalid model")

