import numpy as np


class Perceptron:

    def __init__(self, n_iter=100):
        self.n_iter = n_iter
        self.w = np.empty(shape=0)

    def fit(self, X_train, y_train):
        # 1. Pick random weights:
        self.w = np.random.randn(len(X_train[0]))
        # 2. Iterate for n_iter times:
        i = 0
        while i < self.n_iter:
            # 3. Perform w(T)X to get predictions vector:
            y_pred = np.sign(np.dot(X_train, self.w))
            # 4. On predictions vector, pick a misclassified point:
            converged = True
            for y_hat_i, y_i, x_i in zip(y_pred, y_train, X_train):
                if y_hat_i != y_i:
                    # Correct weight for that point and break the loop:
                    converged = False
                    self.w = self.w + np.dot(x_i, y_i)
                    break
            # 5. If no misclassified points found - stop:
            if converged:
                break
            # 6. Iterate further:
            i += 1

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

    def score(self, X_test, y_test):
        pass


p = Perceptron()
X = [[1, 1], [1, 2], [2, 2], [4, 1], [5, 2], [3, 4]]
y = [1, 1, 1, -1, -1, 1]

p.fit(X, y)
print(p.w)

import sklearn.linear_model

sk_p = sklearn.linear_model.Perceptron(max_iter=100)
sk_p.fit(X, y)
print(sk_p.coef_)

print(p.predict(X))
print(sk_p.predict(X))

