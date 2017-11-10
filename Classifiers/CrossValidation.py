"""
Module for cross-validation
"""
import numpy as np

def KFold(X_data, k=2):
    """
    Calculates start and end indices of test set for each fold. To use it, just define a
    set as a test set when it has indices returned from this function, and make the
    rest of dataset a train set. Example:
        indices = KFold(X, 2)
        for pair in indices:
            start, end = pair[0], pair[1]
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.delete(X, np.s_[start:end], 0)
            y_train = np.delete(y, np.s_[start:end], 0)
            # Now fit model with train set.. Calculater error with test set..
    :param X_data: numpy array of observations
    :param k: desired amount of samples in test set
    :return: list of pairs [start_index, end_index] - indices of train set at every step
    """
    for i in range(0, len(X_data), k):
        yield [i, (i + k)]
        

def leave_one_out(X_data):
    """
    Calculates start and end indices of test set for each fold. To use it, just define a
    set as a test set when it has indices returned from this function, and make the
    rest of dataset a train set. Example:
        indices = KFold(X, 2)
        for pair in indices:
            start, end = pair[0], pair[1]
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.delete(X, np.s_[start:end], 0)
            y_train = np.delete(y, np.s_[start:end], 0)
            # Now fit model with train set.. Calculater error with test set..
    :param X_data: numpy array of observations
    :return: list of pairs [start_index, end_index] - indices of train set at every step
    """
    return KFold(X_data, 1)    

# Test data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])     
 
"""
#Test for K-FOLD        
indices = KFold(X, 2)
for pair in indices:
    start, end = pair[0], pair[1]
    X_test, y_test = X[start:end], y[start:end]
    X_train = np.delete(X, np.s_[start:end], 0)
    y_train = np.delete(y, np.s_[start:end], 0)
    print("\n============================\n", X_test, y_test,"\n",X_train, y_train)
"""
"""
#Test for LEAVE ONE OUT
indices = leave_one_out(X)
for pair in indices:
    start, end = pair[0], pair[1]
    X_test, y_test = X[start:end], y[start:end]
    X_train = np.delete(X, np.s_[start:end], 0)
    y_train = np.delete(y, np.s_[start:end], 0)
    print("\n============================\n", X_test, y_test,"\n",X_train, y_train)
"""
