import numpy as np

class LogisticRegression:

    def __init__(self):
        # Simply default parameters
        self.W = np.ones(shape=(3,1))
        self.b = 0.4
        self.num_labels = 2
        self.max_iterations = 100 
    
    
    def train(self, X_train, y_train):
        # 1. Determine amount of LABELS and FEATURES
        self.num_labels = len(set(y_train))
        num_features = X_train.shape[1] # ATTENTION this line can lead to bugs!
        # 2. Initialize weights and bias
        if self.num_labels == 2:
            print("DOES NOT WORK WITH 2-CLASS PROBLEMS FOR NOW")
            return
            #self.W = np.ones(shape=(num_features))
            #self.b = np.ones(shape=(1))
        else:
            self.draw_weights_and_bias(0, 0.1, self.num_labels, num_features)    
        #print(self.W, self.W.shape, self.b, self.b.shape)
        # 4. Train!
        for X, y in zip(X_train, y_train):
            #print(X, y)
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
    
    
    def cross_entropy(self, S, L):
        """
        >>> As it is told in google's course <<<
        Calculates cross-entropy between S and L

        :param S: vector of Softmax values - S(y)
        :param L: vector of true labels
        :return: returns cross-entropy between S and L
        """
        res = 0
        for s,l in zip(S, L):
            res += l*np.log10(s)
        return -1 * res
    
    def loss(self, X_train, y_train, verbose=0):
        """Calculates average cross-entropy over a training set"""
        """
        FIXME PROBLEM:
            In case of num_labels is 2 (only 2 classes, so the problem is not multinominal)
            softmax here is set to return WX+b, which depending on X can be NEGATIVE. It gets passed to
            log10 in cross entropy calculation and it results in ERROR.
        SOLUTIONS:
            * Do not use for 2-class problems? For 2-class use another algoritm?
            * Count a 2-class problem as multinominal?
                Meaning, instead of "y = 0.13" use "y = (y1, y2) ---> S(y) = (0, 1) ---> meaning 
                it's 2nd class and not the 1st one"
        """
        entropy = 0
        if verbose:
            print("Weights:\n", self.W, "\nBias:\n", self.b)
        for Xi, yi in zip(X_train, y_train):
            # 1. Y = W*X + b
            Y = np.dot(self.W, Xi) + self.b
            # 2. S(Y) if needed
            # FIXME after testing delete the following line:
            self.num_labels = 3         
            if self.num_labels == 2:
                S = Y # Otherwise softmax will ALWAYS yield 1!
                # TODO USE SOMETHING OTHER, I think! Not WX+b or NOT cross-entropy
            else:
                S = self.softmax(Y)
            # 3. Add cross-entropy to counter:
            entropy += self.cross_entropy(S, [yi])
            if verbose:
                print("\n=============================")
                print("Xi and yi:", Xi, yi)
                print("WX + b =", Y)
                print("Softmax result:", S)
                print("Entropy change on this step:", self.cross_entropy(S, [yi]))
                print("Total entropy as of now:", entropy)
        return entropy / len(y_train)
    
    
    def encode_one_hot(self, vector):
        """Returns a one-hot encoded vector derived from a specified vector."""
        max_v = max(vector)
        return np.array(list(map(lambda x: 1 if x==max_v else 0, vector)))
    
              
    def check_predictions(self, test_X, test_Y):
        pass
    
    def draw_weights_and_bias(self, mu, sigma, num_labels, num_features):
        """Draws W and bias from Gaussian distribution"""
        self.W = np.random.normal(mu, sigma, size=(num_labels, num_features))
        self.b = np.random.normal(mu, sigma, size=(num_labels))


# 0. Create data and classifier
# AND relation
"""
X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([0,1,0,0,0], dtype='float64')
y_test  = np.array([0,0,0], dtype='float64')
"""

# OR relation
"""
X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([1,1,1,1,0], dtype='float64')
y_test  = np.array([1,1,1], dtype='float64')
"""
# MULTINOMINAL set
X_train = np.array([[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,0,0]], dtype='float64')
X_test  = np.array([[0,1,0],[1,0,0],[0,1,1]], dtype='float64')
y_train = np.array([0,1,0,2,0], dtype='float64')
y_test  = np.array([0,0,2], dtype='float64')
        
clf = LogisticRegression()

# 1. Set weights and bias:
#clf.W = np.array([0.5, -2, 0.5], dtype = 'float64')
#clf.b = np.array([0.2], dtype = 'float64')
clf.W = np.ones(shape=(3, 3))
clf.b = np.array([1, 1, 1], dtype = 'float64')

# 1b. Little sample test of work:
#X = np.array([1, 2, 3], dtype = 'float64')
#print(clf.predict(X))
#print("W, b and shapes:\n", clf.W, clf.W.shape, clf.b, clf.b.shape)

# 2. Train:
#y_train = np.array([[0],[1],[0],[0],[0]], dtype='float64')
clf.train(X_train, y_train)

# 3. Test:


"""
=== Various little tests ===
"""
# Cross entropy
#S = np.array([1, 0.2, 0.1], dtype = 'float64')
#L = np.array([1, 0, 0], dtype = 'float64')
#print(clf.cross_entropy(S, L))

#print(clf.loss(X_train, y_train, verbose=1))
