from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split 

def train(data, X_cols, y_cols, type, sample_weight):
    """
        Summary of function.

        Description:
        -----------------------
        This function returns a model so the users can use to predict other data 

        Parameters (json file):
        ------------------------
        data : csv
            A csv file that have features and targets as keys and a list of values as values
        X_cols: list
            List of the features of the data
        y_cols: list
            List of the targets of the data
        type: string
            Type of Naive Bayes model
        sample_weight: list
            A list of weights for each features 


        Returns (json file):
        -------------------------
        model : scikit-learn model
            a model that can predict other unknown value
    """

    if type.lower() == "gaussian naive bayes":
        model = GaussianNB()
    elif type.lower() == "multinomial naive bayes":
        model = MultinomialNB()
    elif type.lower() == "bernoulli naive bayes":
        model = BernoulliNB()

    X_train, X_test, y_train, y_test = train_test_split(data[X_cols], data[y_cols], test_size=0.2, random_state=42)

    model.fit(X_train, X_test, sample_weight=sample_weight)

    return model

