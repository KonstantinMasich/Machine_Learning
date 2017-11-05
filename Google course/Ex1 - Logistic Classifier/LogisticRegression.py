import numpy as np

class LogisticRegression:

    def __init__(self):
        # Simply default parameters
        self.W = np.ones(shape=(3,1))
        self.b = 0.4
        self.num_labels = 2
        self.max_iterations = 100 
    
    
    def train(self, X_train, y_train):
        # 1. Determine amount of LABELS
        self.num_labels = len(set(y_train))
        num_features = X_train.shape[1] # ATTENTION this line can lead to bugs!
        # 2. Determine shapes of W and bias: TODO - set only shapes, not "ones"
        if self.num_labels == 2:
            self.W = np.ones(shape=(num_features))
            self.b = np.ones(shape=(1))
        else:
            self.W = np.ones(shape=(self.num_labels, num_features))
            self.b = np.ones(shape=(self.num_labels, 1))
        print(self.W.shape, self.b.shape)
        # 3. TODO Set good starting W and b:
        pass
        # 4. Train!
        for X, y in zip(X_train, y_train):
            print(X, y)
            i = 0
            while i < self.max_iterations:
                # do training stuff
                i += 1
        print("All done!")
    
        
    def predict(self, X):
        if self.num_labels==2:
            return np.dot(self.W, X) + self.b
        # 1. Y = W*X + b
        Y = np.dot(self.W, X) + self.b
        # 2. S(Y)
        S = self.softmax(Y)
        # 3. One-hot encoding:
        H = self.encode_one_hot(S) # "H" for one-[H]ot
        return H
    
    
    def softmax(self, vector_Y):
        """Returns softmax probabilities for vector V"""
        l = []
        for y in vector_Y:
            l.append(np.exp(y) / sum(list(map(lambda x: np.exp(x), vector_Y))))
        return np.array(l)
    
    
    def encode_one_hot(self, vector):
        """Returns a one-hot encoded vector derived from a specified vector."""
        max_v = max(vector)
        print("Max is:",max_v)
        return np.array(list(map(lambda x: 1 if x==max_v else 0, vector)))
    
              
    def check_predictions(self, test_X, test_Y):
        pass
    
          
    def foo(self):
        print(self.W)
        
# 0. Create data and classifier
# AND relation
X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([0,1,0,0,0], dtype='float64')
y_test  = np.array([0,0,0], dtype='float64')
        
clf = LogisticRegression()

# 1. Set weights and bias:
W = np.array([0.5, -2, 0.5], dtype = 'float64')
clf.W = W
clf.b = np.array([0.2], dtype = 'float64')
print(clf.W, clf.W.shape, clf.b, clf.b.shape)

# 1b. Little sample test of work:
X = np.array([1, 2, 3], dtype = 'float64')
print(clf.predict(X))

# 2. Train:
clf.train(X_train, y_train)

# 3. Test:

