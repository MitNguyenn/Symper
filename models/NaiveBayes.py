from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score

import uuid
from sklearn.externals import joblib


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
            model_type: string (gaussian/multinomial/bernoulli)
                Type of Naive Bayes model used
            priors=None: array (Shape = target.shape)
                Prior probabilities of the classes
            alpha=1e-9: float
                A number create stability for calculating


        Returns (json file):
        -------------------------
        model_id : string
            The id of the model that can predict other unknown value
        evaluation: dictionary
            accuracy: float
                The accuracy of the model
            precision: float
                The preciion of model

    """
    test_size = params["test_size"]
    type = params["model_type"]
    alpha = params["alpha"]


    X = data.drop(columns=target_columns)
    y = data[target_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if type.lower() == "gaussian":
        model = GaussianNB(var_smoothing=alpha, priors=None)
    elif type.lower() == "multinomial":
        model = MultinomialNB(alpha=alpha, priors=None)
    elif type.lower() == "bernoulli":
        model = BernoulliNB(alpha=alpha)


    model.fit(X_train, y_train)

    evaluation = {}

    evaluation["accuracy"] = accuracy_score(X_test, y_test)
    evaluation["precision"] = precision_score(X_test, y_test)

    model_id= str(uuid.uuid4())
    joblib.dump(model, f"save/{model_id}.pkl")

    return model_id, evaluation

