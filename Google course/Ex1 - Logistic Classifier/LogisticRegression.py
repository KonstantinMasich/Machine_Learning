import numpy as np

class LogisticRegression:
    
    def __init__(self, shape=(3,1), bias=0.0, verbose=0):
        self.shape = shape
        self.W = np.ones(shape)
        self.b = bias
        self.verbose = verbose
        if self.verbose:
            print("INIT PARAMETERS\nWeights:\n", self.W)
            print("Bias = ", self.b)
        
        
    def set_weights(self, W):
        """Sets Weights to a specified matrix / vector."""
        self.W = W
    
    
    def set_bias(self, bias):
        """Sets Bias to a specified value."""
        self.b = bias
    
    
    def train(self, train_X, train_Y):
        pass
        
        
    def predict(self, X):
        # 1. Y = W*X + b
        Y = np.dot(X, self.W)
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
        
        
a = LogisticRegression(verbose=0)
#a.foo()
l = [2.0, 1.0, 0.1]
print("l:",l)
S = a.softmax(np.array(l))
print(a.encode_one_hot(S))
# 1. Set weights:

# 2. Train:

# 3. Test:
X = np.array(l)
print(a.predict(X))
