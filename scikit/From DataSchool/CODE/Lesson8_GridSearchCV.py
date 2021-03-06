
# =========================================================================== #
# ================= Choosing hyperparameters GridSearchCV =================== #
# =========================================================================== #

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 0. Loading data:
X = load_iris().data
y = load_iris().target

# 1. Creating a model:
knn = KNeighborsClassifier()

# 2. Creating a parameter grid:
k_range = range(1, 31)
param_grid = {'n_neighbors': k_range}
print("Parameter grid:", param_grid)

# So parameter K will be checked within [1, 30] interval

# 3. Instantiating the grid:
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# NOTE that we do not pass X or y to grid search CV function

# 4. Fitting the grid with data:
grid.fit(X, y)

# 5. Viewing the complete results:
for key, val in grid.cv_results_.items():
    print(key, ":\n", val, "\n-----------------------")

# 6. Viewing the best score, parameters and estimator (model):
print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)
print("Best model:", grid.best_estimator_, "\n-----------------------")

# 7. In case we need to check >1 parameters simultaneously:
print("Now checking best N_NEIGHBORS and WEIGHTS params:")
k_range = range(1, 31)
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'weights': weight_options}
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)
print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_, "\n-----------------------")

# 8. After we pick the best model and hyperparameters,
# we should train it on the WHOLE available dataset; otherwise
# we would miss out valuable training data!
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)

# 9a. We now can check our classifier on out-of-sample data:
print("For some unseen input we get:", knn.predict([[3, 5, 4, 2]]))
# 9b. And we can do the same using GridSearchCV OBJECT:
print("The same is:", grid.predict([[3, 5, 4, 2]]), "\n-----------------------")
