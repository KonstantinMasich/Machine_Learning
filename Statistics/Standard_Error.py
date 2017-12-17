"""
====================================================================
 * This is to check calculation of Standard Error of coefficients.
====================================================================

R CODE TO CHECK RESULTS:
mydata = data.frame(
    TV = c(1, 4, 5, 3, 5),
    Radio = c(2, 3, 1, 2, 6),
    Newspapers = c(2, 3, 4, 1, 7),
    sales = c(2, 3, 2, 4, 7)
)
model = lm(sales ~ TV + Radio + Newspapers, data = mydata)
summary(model)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = [[1, 2, 2],[4, 3, 3],[5, 1, 4],[3, 2, 1],[5, 6, 7]]
X_mod = [[1, 1, 2, 2], [1, 4, 3, 3], [1, 5, 1, 4], [1, 3, 2, 1], [1, 5, 6, 7]]
y = [2, 3, 2, 4, 7]

model = LinearRegression()
model.fit(X, y)
print("Coef:", model.coef_, "Bias:", model.intercept_)
y_pred = model.predict(X)
# RSS and REGRESSION VARIANCE
RSS = sum((y_pred-y)**2)
variance = RSS/ (len(y)-len(X[0])-1)
print("Unbiased estimate of regression variance:", variance)
# L matrix
X = np.array(X)
X_mod = np.array(X_mod)
L = np.linalg.inv(np.dot(X_mod.T, X_mod))
print("L matrix:\n", L, "\n","-"*40)
# b0 and b1
b0_err = np.sqrt(variance * L[0][0])
b1_err = np.sqrt(variance * L[1][1])
b2_err = np.sqrt(variance * L[2][2])
print("Standard errors:", b0_err, b1_err, b2_err)
