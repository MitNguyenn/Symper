from Symper.tools import Model

import pandas as pd

def main():
    ...

def create_model(json):
    """
        Summary of function.

        Parameters (json file):
        ------------------------
        model_type: string
            Name of the chosen model
        data : json
            A json file that have features and targets as keys and a list of values as values
        target: list
            a list of targeted values that is needed to be predicted
        parameter: list
            A list of parameter for the model  


        Returns (json file):
        -------------------------
        model : scikit-learn model
            a model that can predict other unknown value
    """
        
    data = pd.read_json(json["data"])
    target = json["target"]
    parameter = pd.read_json(json["parameter"])
    model_type =json["model_type"]

        

    model = Model(type=model_type, X_cols=list(set(data.columns)-set(target)), y_cols=target, parameter=parameter)

    return model


if __name__ == "__main__":
    main()