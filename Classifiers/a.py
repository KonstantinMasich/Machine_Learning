"""
http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
"""

#===========================================================#
#===========================================================#
#= THIS IS AN EXAMPLE OF LOGISTIC REGRESSION, SEE ONE NOTE =#
#===========================================================#
#===========================================================#


import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Turn down for faster convergence
t0 = time.time()
train_samples = 5000


# 1. Here we fetch the builtin MNIST data.
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
# After this step:
#   X - is 2D ndarray of shape (70000, 784), which means it contains 70.000
#       pictures, each one is 28x28pxl (=784pxl)
#   y - is 1D ndarray of shape (70000,) containing labels 0-9, so every row
#       in y contains info about what digit is depicted in the picture
#       in corresponding row of X


# 2. This is where we shuffle the data: we create a random permutation and
# shuffle X and y "equally" according to this permutation. Notice that rows
# of X and y are still corresponding with each other: if X[123] became
# X[413] after shuffling, y[123] will also become y[413], so each label is
# still assigned to "its" picture.
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
# After reshaping, X and y still have their shapes of (7000, 784) and (7000,)

# 3. Here we split data in train and test sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)
# After this step:
#   X_train - has shape(5000, 784); it has 5.000 train samples
#   X_test  - has shape(10000,784); it has 10.000 test samples
#   y_train - has shape(5000,); it has 5000 train labels
#   y_Test  - has shape(10000,);it has 10.000 test labes
# So we basically get all the 70.000 samples and split it: 10.000 for test set,
# and <train_samples> (=5000 by default) for train set.

# 4. We preprocess the data by scaling it (zero mean / unit variance), see
# documentantion on StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_train is still (5000, 784) and X_test is still (10000, 784)

# 5. This is classifier training itself:
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
# See documentation on LogisticRegression
print(clf.coef_, "\n", clf.coef_.shape) # Weights matrix
print(clf.intercept_, "\n", clf.intercept_.shape) # Bias
