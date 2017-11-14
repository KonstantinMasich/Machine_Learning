import numpy as np
import operator

class KNN:


    def __init__(self, X, y, n_neighbors=5):
        self.X = X
        self.y = y
        self.K = n_neighbors
        
        
    def predict(self, X):
        # 1.Searching for K nearest neihghbors
        distances = []
        for observation in self.X:
            distances.append(np.linalg.norm(observation - X))
        dist_dictionary = dict(enumerate(distances))
        # 2. Sorting distances and getting K nearest neighbors EXCEPT for dist=0.0 of course:
        dist_tuples = sorted(dist_dictionary.items(), key=operator.itemgetter(1))[1:self.K+1]
        # 3. Finding out the most common class within those K nearest neighbors:
        indices = [x[0] for x in dist_tuples]
        y_s = []
        for index in indices:
            y_s.append(self.y[index])
        return self.__most_common(y_s)
        
        
    def test(self, X_test, y_test):
        matches = 0
        for x, y in zip(X_test, y_test):
            if self.predict(x) == y:
                matches +=1
        return matches/len(y_test)
    
    
    def __most_common(self, l):
        from collections import Counter
        freq = (el for el in l)
        c = Counter(l)
        try:
            if (c.most_common(2)[0][1] == c.most_common(2)[1][1]):
                return None
        except IndexError:
            return c.most_common(2)[0][0]

"""
TESTING
# 1. Obtaining dataset:
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. Shuffling dataset:
from sklearn.utils import check_random_state
random_state = check_random_state(123)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
#print(y)

# 3. Trying to predict:
import time
t0 = time.time()

for i in range(1, 13):
    knn = KNN(X, y, n_neighbors=i)
    print("i=",i,"; score =", knn.test(X, y))
t1 = time.time()
print("Time elapsed during predicting:", t1-t0)



v1 = [0.1, 2.5, 3.6]
v2 = [1.1, 7.8, 4.4]
v1 = np.array(v1)
v2 = np.array(v2)

t0 = time.time()
#print(np.sqrt(np.sum(np.square(v1-v2))))
for i in range(1, 100000):
    dist = np.sqrt(np.sum(np.square(v1-v2)))
t1 = time.time()
print("Time elapsed:", t1-t0)


t0 = time.time()
#print("Numpy distance is: ", np.linalg.norm(v1 - v2))
for i in range(1, 100000):
    dist = np.linalg.norm(v1 - v2)
t1 = time.time()
print("Time elapsed:", t1-t0)
"""
