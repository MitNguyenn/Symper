from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train(data, target_columns, params):
    """
        Summary of function.
 
        Description:
        -----------------------
        This function returns a model so the users can use to predict other data
 
        Parameters (json file):
        ------------------------
        data: pd.DataFrame
            A DataFrame consists of float/int
        target_columns: array
            An array of string containing the target names 
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data
            penalty="l2": string ("l1", "l2", "elasticnet", "None")
                The normalization of the penalty
            tol=1e-4: float 
                Tolerance for stopping criteria
            C=1.0: float (must be positive)
                Inverse of regularization strength: The smaller the number, the stronger the regularization
            fit_intercept=True: bool
                Weather to have bias in the model or not 

        Returns (json file):
        -------------------------
        model : sklearn.linear_model._base.LinearRegression
            a model that can predict other unknown value
        evaluation: dictionary
            accuracy:
                The accuracy of the model
            
    """

    return 0