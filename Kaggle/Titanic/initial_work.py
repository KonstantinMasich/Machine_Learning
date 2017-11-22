import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 1. Load data into train and test files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

"""
2. What features seem to be excessive? For now I would say it's:
 - name
 - cabin (it's actually an important feature, but 687 out of 891 samples
   have "NaN" in this category, so it's not that informative to us)
 - ticket (ticket number is not that important is it?)
 - Maybe: embarkation port? 
 - Passenger ID is for ID purposes only, do NOT include it into model
   What really is important is:
   - sex (females more likely to survive?)
   - age (children more likely to survive?)
   - Pclass (rich guys more likely to survive?)
"""
# 2. Replace all "male" and "female" with 0 and 1 respectively
# and C, Q and S with 0, 1, 2 as well:
train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], 1)
mapping = {'male': 0, 'female': 1, 'C': 0, 'Q': 1, 'S': 2}
train_df = train_df.replace({'Sex': mapping, 'Embarked': mapping})
test_df = test_df.replace({'Sex': mapping})
# print(train_df.head())

# 3. Create train and test sets
y = train_df['Survived']
X = train_df.drop('Survived', 1)
#print(y.head())
#print(X.head())
X = X.fillna(0)  # REPLACE ALL NaN with 0 -- needed or not?

#sns.pairplot(train_df, x_vars=['Pclass'], y_vars='Survived', size=7, aspect=0.7, kind='reg')
#plt.show()


# 4. Let's try KNN for this problem and see how well does it perform.
knn = KNeighborsClassifier(n_neighbors=8, weights='distance')
scores = cross_val_score(knn, X, y, scoring='accuracy', cv=15)
print("Mean cross-validated score is:", scores.mean())

"""
# 5. Let's use LogisticRegression for the problem:
log_regression = LogisticRegression()
scores = cross_val_score(log_regression, X, y, scoring='accuracy', cv=10)
#print("Cross validated scores are:", scores)
#print("Mean is:", scores.mean())

# 6. Now let's perform grid search CV for basic parameters:
knn = KNeighborsClassifier()
k_range = range(1, 31)
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'weights': weight_options}
grid = GridSearchCV(knn, param_grid, cv=15, scoring='accuracy')
grid.fit(X, y)
print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)
print("Best model:", grid.best_estimator_, "\n---------------")
# Best score = 0.7351, best params: n_neighbors = 8, weights = 'distance'
"""
