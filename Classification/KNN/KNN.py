import numpy as np
from operator import itemgetter


class KNN:
    def __init__(self, n_neighbors=5):
        self.X = None
        self.y = None
        self.K = n_neighbors

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, usr_X):
        # 0. Do we have matrix X as an input, or just single vector x?
        if np.array(usr_X).ndim == 1: # vector
            X = np.array([usr_X[:]])
        else:                         # matrix
            X = np.array(usr_X[:])
        res = []
        # Loop through each point in X (or once if provided vector x)
        for x in X:
            # 1. First of all we create a dictionary of indices and distances, where
            #    for each observation at X (or for vector x):
            #    {0: d_0} - meaning distance between x and self.X[0] is d_0
            #    {1: d_1} - meaning distance between x and self.X[1] is d_1
            #    {2: d_2} - meaning distance between x and self.X[2] is d_2
            #    etc.
            distances = []
            for observation in self.X:
                distances.append(np.linalg.norm(observation - x))
            dist_dictionary = dict(enumerate(distances))
            # 2. Sort distances and get K nearest neighbors EXCEPT for dist=0.0 of course:
            dist_tuples = sorted(dist_dictionary.items(), key=itemgetter(1))[1:self.K + 1]
            # Now it's list of K tuples like [(4, 2.17),(2, 12.5),(18, 34.5)] showing
            # indices of K closest neighbors and distances to those neighbors.
            # 3. Find out the most common class within those K nearest neighbors:
            y_s = list(map(lambda index: self.y[index], [elem[0] for elem in dist_tuples]))
            # y_s is a list that holds K most common classes, like [0, 3, 0]
            res.append(max(set(y_s), key=y_s.count))
        return res

    def score(self, X_test, y_test):
        # 1. Get a prediction:
        y_pred = self.predict(X_test)
        # 2. Compare results:
        match_counter = 0
        for y_pred_i, y_i in zip(y_pred, y_test):
            if y_pred_i == y_i:
                match_counter += 1
        return match_counter / len(y_test)
