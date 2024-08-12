from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
            fit_intercept=True: bool
                Weather to have bias in the model or not
            positive=True: bool
                Weather to set bias to always be positive or not
        

        Returns (json):
        -------------------------
        model : string
            The id of a model that can predict other unknown value
        evaluation: dictionary
            MSE:
                Mean Squared Error loss of the model. The smaller, the better
    """

    return 0