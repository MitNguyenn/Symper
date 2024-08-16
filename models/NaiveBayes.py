import os

import uuid
import joblib

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score

def train(data, target_columns, params):
    """
        Summary of the function.

        Description:
        -----------------------
        This function returns a model that users can utilize to predict other data.
    
        Parameters:
        ------------------------
        data: pd.DataFrame
            A DataFrame consisting of floats/integers.
        target_columns: array
            An array of strings containing the target names.
        params: dictionary
            test_size: float (between 0 and 1)
                The percentage of validation data taken from the data.
            model_type: string (gaussian/multinomial/bernoulli)
                The type of Naive Bayes model to be used.
            priors: array, optional (default=None) (shape = target.shape)
                Prior probabilities of the classes.
            alpha: float, optional (default=1e-9)
                A smoothing parameter to create stability for calculations.

        Returns:
        -------------------------
        model_id: string
            The ID of the model that can predict other unknown values.
        evaluation: dictionary
            accuracy: float
                The accuracy of the model.
            precision: float
                The precision of the model.
    """
    
    test_size = params.get('test_size', 0.2)
    type = params.get('type')
    alpha = params.get('alpha', 1e-9)
    priors = params.get('priors', None)


    X = data.drop(columns=target_columns)
    y = data[target_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if type.lower() == "gaussian":
        model = GaussianNB(var_smoothing=alpha, priors=priors)
    elif type.lower() == "multinomial":
        for col in y.columns:
            if len(y[col].unique()) != 2:
                raise ValueError("Targets must only have 2 unique values") 
        model = MultinomialNB(alpha=alpha, class_prior=priors)
    elif type.lower() == "bernoulli":
        model = BernoulliNB(alpha=alpha, class_prior=priors)
    else:
        raise ValueError("Invalid Naive Bayes model type")


    model.fit(X_train, y_train)

    evaluation = {}

    evaluation["accuracy"] = accuracy_score(model.predict(X_test), y_test)
    evaluation["precision"] = precision_score(model.predict(X_test), y_test)

    model_id = str(uuid.uuid4())
    if not os.path.exists("save"):
        os.makedirs("save")
    joblib.dump(model, f"save/{y.columns.to_list()}`~{model_id}.pkl")
    

    return model_id, evaluation

