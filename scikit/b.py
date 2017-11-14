import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. We load data:
iris = load_iris()
X = iris.data
y = iris.target

# 2. Now we shuffle the data randomly:
from sklearn.utils import check_random_state
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

# 3. And then we use CV on both models to find which one is the best:
# KNN with 10-fold CV:
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
# LR with 10-fold CV:
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
#Results are: 0.96 for KNN, 0.9533333333 for LR

# 4. Let's do it in a loop!
for i in range(0, 50):
    print ("========== random seed =",i," ==========")
    random_state = check_random_state(i)
    permutation = random_state.permutation(X.shape[0])
    X, y = X[permutation], y[permutation]
    X = X.reshape((X.shape[0], -1))
    knn = KNeighborsClassifier(n_neighbors=20)
    logreg = LogisticRegression()
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean(), \
        cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
