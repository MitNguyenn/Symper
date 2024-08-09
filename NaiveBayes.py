from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split 
import pandas as pd

def train(data, X_cols, y_cols, type, sample_weight, alpha):
    """
        Summary of function.

        Description:
        -----------------------
        This function returns a model so the users can use to predict other data 

        Parameters (json file):
        ------------------------
        data : pd.DataFrame
            A DataFrame containing the data
        X_cols: list
            List of the features of the data
        y_cols: list of string
            List of the targets of the data
        type: string
            Type of Naive Bayes model
        sample_weight: list of float
            A list of weights for each features 
        alpha: float/ int
            A number to make training more stable


        Returns (json file):
        -------------------------
        model : scikit-learn model
            a model that can predict other unknown value
    """

    if type.lower() == "gaussian naive bayes":
        model = GaussianNB(var_smoothing=alpha)
    elif type.lower() == "multinomial naive bayes":
        model = MultinomialNB(alpha=alpha)
    elif type.lower() == "bernoulli naive bayes":
        model = BernoulliNB(alpha=alpha)


    model.fit(data[X_cols], data[y_cols], sample_weight=sample_weight)

    return model

