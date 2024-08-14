from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

import uuid
import joblib

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
                Whether to have bias in the model or not 

        Returns (json file):
        -------------------------
        model : string
            The id of a model that can predict other unknown value
        evaluation: dictionary
            accuracy:
                The accuracy of the model
            precision:
                The precision of the model
            
    """

    X = data.drop(target_columns)
    y = data[target_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.get('test_size', 0.2))

    model = LogisticRegression(
        penalty=params.get('penalty', 'l2'),
        tol=params.get('tol', 1e-4),
        C=params.get('C', 1.0),
        fit_intercept=params.get('fit_intercept', True)
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    evaluation = {
        'accuracy': accuracy,
        'precision': precision
    }
    
    model_id = str(uuid.uuid4())
    
    joblib.dump(model, f"save/{model_id}.pkl")
    
    return model_id, evaluation
