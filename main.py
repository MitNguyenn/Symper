from Symper.tools import Model

import pandas as pd

def main():
    ...

def create_model(json):
    """
        Summary of function.

        Description:
        -----------------------
        This function returns a model so the users can use to predict other data 

        Parameters (json file):
        ------------------------
        model: string
            The type of model using
        data : csv
            A csv file that have features and targets as keys and a list of values as values
        target: list
            a list of targeted values that is needed to be predicted
        parameter: list
            A list of parameter for the model  


        Returns (json file):
        -------------------------
        model : scikit-learn model
            a model that can predict other unknown value
    """
    #TODO: implement the function 

def predict(json):
    """
        Summary of function.

        Description:
        --------------------------------
        Returns the prediction given a model and its data

        Parameters (json file):
        ---------------------------------------
        model: scikit-learn model
            The chosen model used for predicting
        data: csv
            A file consists of features that needed to be predicted
    

        Returns (json file):
        -----------------------------
        prediction: csv file
            The predicted value for the given data
    """
    #TODO: implement the function
    

if __name__ == "__main__":
    main()