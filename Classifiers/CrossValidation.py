"""
Module for cross-validation
"""
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

def KFold(X_data, y_data, k=2):
    # 1. Create split sets:
    X_set, y_set = __splitter(X_data, y_data, k)
    # 2. For each subset of split sets:
    X_res = [], y_res = []
    for X, y in X_set, y_set:
        
        
    return X_set, y_set

        
def __splitter(X_data, y_data, k=2):
    for i in xrange(0, len(X_data), k):
        #print X_data[i:(i + k)], y_data[i:(i + k)]
        yield X_data[i:(i + k)], y_data[i:(i + k)]

kf = KFold(X, y, 2)
for i in kf:
    print (i)

class Cross_validator:
    """
    This class assumes that all the input is correct and X_data and Y_data 
    are evenly sized arrays.
    """
    
    def __init__(self, X_data, y_data, validation_type='k-fold', k=-1):
        self.types = ['k-fold', 'one_out', 'random_subsets']
        self.X_data = X_data
        self.y_data = y_data
        self.validation_type = validation_type
        self.k = k # Fold size
    
    def cross_validate(self):
        # Illegal validation type
        if self.validation_type not in self.types:
            print("Error: CROSS_VALIDATOR -> illegal validation type specified!")
            return None
        # K-FOLD cross-validation
        if self.validation_type == self.types[0]:
            if self.k == -1: # Amount of folds is not specified
                print("Error: CROSS_VALIDATOR -> amount of folds is not specified!")
                return None
            X_set = []
            y_set = []    
            for i in xrange(0, len(self.X_data), self.k):
                X_set.append(self.X_data[i:i + self.k])
                y_set.append(self.y_data[i:i + self.k])
            print(X_set, y_set)
                
    """
    def __divide_lists(self, X_data, y_data, k):
        for i in xrange(0, len(X_data), k):
            yield X_data[i:i + k], y_data[i:i + k]
    """    
"""        
X = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
y = [['A'],['B'],['C'],['D']]        
cv = Cross_validator(X,y,k=2)
cv.cross_validate()
print ("Hello")
"""
