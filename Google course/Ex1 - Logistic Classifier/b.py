import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Let us try to train a regression model on a different, simple set.
# OR
"""
X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([1,1,1,1,0], dtype='float64')
y_test  = np.array([1,1,1], dtype='float64')
"""

# AND

X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([0,1,0,0,0], dtype='float64')
y_test  = np.array([0,0,0], dtype='float64')

# XOR
"""
X_train = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]], dtype='float64')
X_test  = np.array([[1,0,1],[1,1,0],[1,1,1]], dtype='float64')
y_train = np.array([0,1,1,0,1], dtype='float64')
y_test  = np.array([0,0,1], dtype='float64')
"""

# 3 LABELS?
X_train = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]], dtype='float64')
X_test  = np.array([[1,0,1],[1,1,0],[1,1,1]], dtype='float64')
y_train = np.array([0,1,1,0,1], dtype='float64')
y_test  = np.array([0,0,1], dtype='float64')


print("Raw arrays:\n",X_train,"\n",y_train,"\n")
train_samples = len(X_train)
# 1. Shuffle data:
random_state = check_random_state(1)
permutation = random_state.permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]
X_train = X_train.reshape((X_train.shape[0], -1))
print("After shuffling:\n",X_train,"\n",y_train,"\n")

# 2. Scale data: NOT NEEDED HERE? yields wrong results. Maybe because of zeros?
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""

# 3. Train model:
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)

# 4. Manually try some examples:
t1 = X_test[0].T
p1 = clf.predict(t1.reshape(1, -1))
print(p1)

for sample in X_test:
    t = sample.T
    print("Prediction for", t, "is:", clf.predict(t.reshape(1, -1)))
    
print("All predictions:\n", clf.predict(X_test))
print("\n==================== STATISTICS =====================\n")
print("Weights:")
print(clf.coef_, "\n") # W
print(clf.coef_.shape) # shape of W
print(clf.intercept_)  # bias
print(clf.intercept_.shape) # shape of bias


