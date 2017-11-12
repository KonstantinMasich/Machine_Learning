from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split to test and validation sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
# Here X and y will continue on as validation set, and test sets will not be
# touched until the very end of experiment.

# 3. On validation set apply cross-validation to KNN model
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, scoring='accuracy', cv=10)
print("Scores are:", scores)
print("Average score is:", scores.mean())

glob_scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    glob_scores.append(cross_val_score(knn, X, y, scoring='accuracy', cv=10).mean())
    print("For K =",k,"neighbors, average score is:", glob_scores[-1])
m = max(glob_scores)
print("Max score is:", m)
max_indices = [i for i, j in enumerate(glob_scores) if j == m]
max_indices = [el+1 for el in max_indices]
print("K numbers corresponding to max scores are: ", max_indices)
