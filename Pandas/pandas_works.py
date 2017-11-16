
# =========================================================================== #
# ============================ Learning Pandas ============================== #
# =========================================================================== #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 4. Now let's check linear regression on this data:
lr = LinearRegression()
lr.fit(X_train, y_train)
print(mean_absolute_error(lr.predict(X_test), y_test))
