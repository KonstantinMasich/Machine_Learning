
# =========================================================================== #
# ============================ Learning Pandas ============================== #
# =========================================================================== #

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. Loading CSV file directly from a URL and saving results:
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
print(data.head(), "\n------------------")  # Displaying 5 top rows
print(data.tail(), "\n------------------")  # Displaying 5 bottom rows
print("Dataframe shape:", data.shape, "\n------------------")

# 2. Visualizing data via SEABORN:
sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')
# plt.show()

# 3. Loading data into X and y sets:
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 4. Now let's check linear regression on this data:
lr = LinearRegression()
lr.fit(X_train, y_train)
print("LR intercept:", lr.intercept_)
print("LR coefficients:", lr.coef_)
y_pred = lr.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)), "\n------------------")

# 5. We see that Newspaper feature is less significant then TV and radio, so maybe
# we can remove it and get a better performance?
feature_cols = ['TV', 'radio']
X = data[feature_cols]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("Without Newspapers feature, RMSE is:", RMSE, "\n------------------")

# 6. Let's make a little loop for excluding the "worst" feature:
features = ['TV', 'radio', 'newspaper']
for excluded_feature in features:
    # Creating features subset
    features_subset = features.copy()
    features_subset.remove(excluded_feature)
    # Creating LR object and measuring error:
    X = data[features_subset]
    y = data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print("On features subset:", features_subset, "RMSE is:", RMSE, "\n------------------")
