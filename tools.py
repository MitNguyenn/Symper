

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

class Model:
    def __init__(self, X_cols, y_cols, type, parameter):
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.type = type
        self.parameter = parameter

        self.trained = False

        if self.type.lower() == "linear regression":
            self.model = LinearRegression()
            print(f"Parameters are: \n - fit_intercept (bool), \n - potitive (bool)")
        elif self.type.lower() == "logistic regression":
            self.model = LogisticRegression()
            print(f"Parameters are: \n - ")
        elif self.type.lower() == "gaussian naive bayes":
            self.model = GaussianNB()
        elif self.type.lower() == "multinomial naive bayes":
            self.model = MultinomialNB()
        elif self.type.lower() == "bernoulli naive bayes":
            self.model = BernoulliNB()


    def train(self, data, sample_weight=None):
        X_train, X_test, y_train, y_test = train_test_split(data[self.X_cols], data[self.y_cols], test_size=0.2, random_state=42)

        self.model.fit(X_train, X_test, sample_weight=sample_weight)

        self.trained = True


    def predict(self, X):
        if not self.trained:
            print("Model is not trained")
            return 0
        
        return self.model.predict(X)
    
    def evaluate(self):
        ...