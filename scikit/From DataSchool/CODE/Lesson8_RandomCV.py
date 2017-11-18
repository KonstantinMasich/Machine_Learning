# =========================================================================== #
# ============== Choosing hyperparameters RandomizedSearchCV ================ #
# =========================================================================== #

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 0. Loading data:
X = load_iris().data
y = load_iris().target

# 1. Creating a model:
knn = KNeighborsClassifier()

# 2. Creating a parameter distributions:
k_range = range(1, 31)
weight_options = ['uniform', 'distance']
param_dist = {'n_neighbors': k_range, 'weights': weight_options}
# NOTE: for continuous parameters create continuous distribution!!!
# i.e. not a list! It'll provide better search.

# 3. Instantiating RandomizedSearch:
random_CV = RandomizedSearchCV(knn,
                               param_dist,
                               cv=10,
                               scoring='accuracy',
                               n_iter=10,
                               random_state=5)

# 4. Fitting random_CV with datA:
random_CV.fit(X, y)

# 5. Viewing scores and best score + params:
print(random_CV.cv_results_)
print("BEST SCORE is:", random_CV.best_score_)
print("This score is achieved with:", random_CV.best_params_)


