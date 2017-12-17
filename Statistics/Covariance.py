"""
===========================================================
* COVARIANCE CALCULATION
===========================================================
"""
import numpy as np
"""
A = np.array([1, 2, 1, -2, 3, 4, -3, 5, -2])
B = np.array([3, -1, -3, -2, 0, 4, 5, 4, -7])
"""

x = np.array([[0, 2], [1, 1], [2, 0]]).T
x = np.array([[0, 2], [1, 1], [2, 0]]).T
# NumPy result:
# Z = np.array([[a, b] for a, b in zip(A, B)]).T
print("NumPy covariance matrix:\n", np.cov(x),"\n","="*40)


# Getting means:
A = np.array([1, 2, 1, -2, 3, 4, -3, 5, -2])
B = np.array([3, -1, -3, -2, 0, 4, 5, 4, -7])

print(np.cov(np.vstack((A, B))))


# A = np.array([0, 1, 2])
# B = np.array([2, 0, 1])

a_mean, b_mean = A.mean(), B.mean()
print("mean A:", a_mean, "; mean B:", b_mean)

print(np.dot((A - a_mean), (B - b_mean)))




