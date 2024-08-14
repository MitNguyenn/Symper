from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

import uuid
import joblib

def train(data, target_columns, params):
    """
        Summary of the function.
    
        Description:
        -----------------------
        This function returns a model that users can utilize to predict other data.
    
        Parameters (JSON file):
        ------------------------
        data: pd.DataFrame
            A DataFrame consisting of floats/integers.
        target_columns: array
            An array of strings containing the target names.
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of data used for validation.
            penalty: string, optional (default="l2") ("l1", "l2", "elasticnet", "None")
                The type of regularization penalty to apply.
            tol: float, optional (default=1e-4)
                Tolerance for stopping criteria.
            C: float, optional (default=1.0, must be positive)
                Inverse of regularization strength: A smaller value indicates stronger regularization.
            fit_intercept: bool, optional (default=True)
                Whether to include an intercept in the model.

        Returns:
        -------------------------
        model: string
            The ID of the model that can predict other unknown values.
        evaluation: dictionary
            accuracy: float
                The accuracy of the model.
            precision: float
                The precision of the model.
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
