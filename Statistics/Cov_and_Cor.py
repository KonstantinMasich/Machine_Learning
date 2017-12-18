# ====================================== #
# COVARIANCE AND CORRELATION CALCULATION #
# ====================================== #
import numpy as np


def covariance(A, B):
    """
    Calculates covariance between parameters A and B
    :param A: array-like, parameter 1
    :param B: array-like, parameter 2
    :return:  covariance between A and B
    """
    _A = np.array(A)
    _B = np.array(B)
    return np.dot((_A - _A.mean()), (_B - _B.mean())) / (len(_A) - 1)


def covariance_matrix(params_list):
    """
    Calculates covariance matrix of parameter list
    :param params_list: array-like, 2D, each row is list of parameter values
    :return: covariance matrix
    """
    cov_matrix = np.zeros(shape=(len(params_list), len(params_list)))
    for i in range(len(params_list)):
        for j in range(len(params_list)):
            cov_matrix[i][j] = covariance(params_list[i], params_list[j])
    return cov_matrix


def correlation_coef(A, B):
    """
    Calculates Pearson correlation coefficient of A and B
    for SAMPLE (not population!)
    :param A: array-like, parameter 1
    :param B: array-like, parameter 2
    :return:  Pearson correlation coefficient r(A,B)
    """
    _A = np.array(A)
    _B = np.array(B)
    num = np.dot(_A, _B) - _A.mean() * _B.mean() * len(_A)
    den = (len(_A) - 1) * np.std(_A, ddof=1) * np.std(_B, ddof=1)
    return num / den


def correlation_matrix(params_list):
    """
    Calculates correlation matrix of parameter list. Calculates for sample,
    not population!
    :param params_list: array-like, 2D, each row is list of parameter values
    :return: correlation matrix
    """
    cor_matrix = np.zeros(shape=(len(params_list), len(params_list)))
    for i in range(len(params_list)):
        for j in range(len(params_list)):
            cor_matrix[i][j] = correlation_coef(params_list[i], params_list[j])
    return cor_matrix


# ================================ TESTING ================================== #
P1 = np.array([1, 2, -1, -2, 3, 4, -3, 5, -2])
P2 = np.array([3, -1, -3, -2, 0, 4, 5, 4, -7])
P3 = np.array([8, -5, -1, 9, 3, -3, 5, 2, -6])
P4 = np.array([0, -1, -3, 2, 4, -4, 1, 7, -3])

params = np.vstack((P1, P2, P3, P4))

print("COVARIATION MATRICES:\n")
print("NumPy covariance matrix:\n", np.cov(params), "\n", "=" * 40)
print("This covariance matrix: \n", covariance_matrix(params))

print("=" * 40, "\nCORRELATION MATRICES:\n")

print("NumPy correlation matrix:\n", np.corrcoef(params), "\n", "=" * 40)
print("This correlation matrix: \n", correlation_matrix(params))
