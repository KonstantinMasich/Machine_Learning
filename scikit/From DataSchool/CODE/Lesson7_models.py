
# =========================================================================== #
# =============== Choosing a model using cross validation =================== #
# =========================================================================== #

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split to test and validation sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
# Here X and y will continue on as validation set, and test sets will not be
# touched until the very end of experiment.

# 3. On validation set apply cross-validation to both models
knn = KNeighborsClassifier(n_neighbors=18)
knn_avg_score = cross_val_score(knn, X, y, scoring='accuracy', cv=10).mean()
log_reg = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
log_reg_avg_score = cross_val_score(log_reg, X, y, scoring='accuracy', cv=10).mean()
print("On validation set, KNN mean accuracy:", knn_avg_score)
print("On validation set, Logistic Regression accuracy:", log_reg_avg_score)
knn.fit(X, y)
log_reg.fit(X, y)
print("On test set, KNN mean accuracy:", knn.score(X_test, y_test))
print("On test set, Logistic Regression mean accuracy:", log_reg.score(X_test, y_test))

"""
# Does cross_val_score actually influence model that was passed to it?
# ANSWER: NO
log_reg = Logistic Regression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
log_reg.fit(X, y)
print("Coefficients of LR before cross-validating:", log_reg.coef_)
avg_score = cross_val_score(log_reg, X, y, scoring='accuracy', cv=10).mean()
print("Average score over X/y:", avg_score)
print("Now coefficients of LR are:", log_reg.coef_)
"""
