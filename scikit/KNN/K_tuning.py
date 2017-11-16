
# =========================================================================== #
# =============== Tuning K and plotting complexity-accuracy ================= #
# =========================================================================== #

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 0. Loading dataset:
X = load_breast_cancer().data
y = load_breast_cancer().target

# 1. Using CROSS-VALIDATION to tune K parameter:
scores = []
for k in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
# Res: k=13 is the best

# 2. Plotting complexity-accuracy graph:
plt.plot(range(1, 30), scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()
# Res: k=13 is the best

# 3. Finally, training model with the best hyperparameters on the WHOLE dataset:
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X, y)
print(knn.score(X, y))


# IRIS DATASET
"""
# 0. Loading dataset
X = load_iris().data
y = load_iris().target

# 1. Using CROSS-VALIDATION to tune K parameter
scores = []
for k in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# 2. Plotting complexity-accuracy graph
plt.plot(range(1, 30), scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()
"""
