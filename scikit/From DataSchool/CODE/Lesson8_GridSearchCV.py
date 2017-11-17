
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
print(grid.cv_results_)
