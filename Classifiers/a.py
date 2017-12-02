import numpy as np
from matplotlib import pyplot as plt
from random import randint
import sklearn.linear_model
from sklearn.model_selection import train_test_split


class Perceptron:

    def __init__(self, n_iter=100, verbose=False):
        self.n_iter = n_iter
        self.X = np.empty(shape=1)
        self.y = np.empty(shape=1)
        self.coef_ = np.empty(shape=1)
        self.intercept_ = 0
        self.verbose = verbose

    def fit(self, X_train, y_train):
        self.X = np.array(X_train)
        self.y = np.array(y_train)
        # 1. Pick random weights while adding w0 for intercept:
        self.coef_ = np.random.randn(len(self.X[0]) + 1)
        # 2. Add x0 as 1 to all rows in X:
        self.X = np.insert(self.X, 0, 1, axis=1)

        # 3. Iterate for n_iter times:
        iter_no = 0
        while iter_no < self.n_iter:
            # Perform w(T)X to get predictions vector:
            y_pred = np.sign(np.dot(self.X, self.coef_))
            # 4. On predictions vector, pick a misclassified point:
            converged = True
            for y_hat_i, y_i, x_i in zip(y_pred, self.y, self.X):
                if y_hat_i != y_i:
                    # Correct weight for that point and break the loop:
                    converged = False
                    self.coef_ = self.coef_ + np.dot(x_i, y_i)
                    break
            # 5. If no misclassified points found - stop:
            if converged:
                if self.verbose:
                    print("Fitting: converged within", iter_no+1, "steps.")
                break
            # 6. Iterate further:iter_no
            iter_no += 1
        self.intercept_ = self.coef_[0]

    def predict(self, X):
        self.X = np.array(X)
        self.X = np.insert(self.X, 0, 1, axis=1)
        res = np.sign(np.dot(self.X, self.coef_))
        return res.astype(int)

    def score(self, X_test, y_test):
        # 1. Get a prediction:
        y_pred = self.predict(X_test)
        # 2. Compare results:
        match_counter = 0
        for y_pred_i, y_i in zip(y_pred, y_test):
            if y_pred_i == y_i:
                match_counter += 1
        return match_counter / len(y_test)


# 1. Creating dataset: X and y
X = []
y = []
for i in range(0, 60):
    X.append([randint(-10, 10), randint(-10, 10)])

# Case of y = x - 2
line_x = np.linspace(-10, 10, 50)
plt.plot(line_x, line_x-2, 'c-')
for index, point in enumerate(X):
    if point[1] > point[0] - 2:
        plt.plot(point[0], point[1], 'ro')
        y.append(1)
    elif point[1] < point[0] - 2:
        plt.plot(point[0], point[1], 'bo')
        y.append(-1)
    else:
        X[index][1] += 1
        plt.plot(point[0], point[1], 'ro')
        y.append(1)
# plt.show()

# Train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=45, test_size=15)

# Prediction
p = Perceptron(verbose=True)
p.fit(X_train, y_train)
print(p.score(X_test, y_test))

sk_p = sklearn.linear_model.Perceptron(max_iter=100)
sk_p.fit(X_train, y_train)
print(sk_p.score(X_test, y_test))
