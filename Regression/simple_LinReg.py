import numpy as np


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None
        pass

    def fit(self, X_train, y_train):
        self.X = np.array(X_train)
        self.y = np.array(y_train)
        # 1. Initialize weights and add w0 (intercept) to weights:
        self.coef_ = np.random.randn(len(self.X[0]) + 1)
        # 2. Add constant x0=1 to all rows in X:
        self.X = np.insert(self.X, 0, [1], axis=1)
        # One step:
        self.coef_ = np.dot(np.linalg.pinv(self.X), self.y)
        self.intercept_ = self.coef_[0]

    def predict(self, usr_X):
        # 0. Do we have matrix X as an input, or just single vector x?
        if np.array(usr_X).ndim == 1:  # vector
            self.X = np.array([usr_X[:]])
        else:  # matrix
            self.X = np.array(usr_X[:])
        # 1. Add constant x0=1 to all rows in X:
        self.X = np.insert(self.X, 0, [1], axis=1)
        # 1. Predict (return dot product):
        return np.dot(self.X, self.coef_)

    def score(self, X_test, y_test, metric='rmse'):
        # 1. Get a prediction:
        y_pred = self.predict(X_test)
        # 2. Calculate score:
        if metric == 'rmse':
            """RMSE"""
            return np.sqrt((np.sum(np.square(y_pred - y_test))) / (len(y_test)))
        elif metric == 'r2':
            "R2"
            y = np.array(y_test)
            return 1 - np.square(np.linalg.norm(y - y_pred)) / np.square(np.linalg.norm(y - y.mean()))
        else:
            return None
