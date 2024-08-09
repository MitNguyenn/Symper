from sklearn.model_selection import train_test_split
from sklearn import linear_model

def test(data, X_cols, y_cols):
    """
        Summary of function.
 
        Description:
        -----------------------
        This function returns a model so the users can use to predict other data
 
        Parameters (json file):
        ------------------------
        data: table
            ///
        X_cols: list of str
            List of the features of the data 
        y_cols: list of str
            List of the targets of the data
        intercept: bool
            Model bias

        Returns (json file):
        -------------------------
        model : scikit-learn model
            a model that can predict other unknown value
    """

    #TODO 1: TRAIN DATA 
    
    X = data[X_cols]
    y = data[y_cols]
    #X_train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = linear_model.LinearRegression()
    model.fit(X_train, X_test)

    #TODO 2: RETURN MODEL
    return model

