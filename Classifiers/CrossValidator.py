"""
Module for cross-validation
"""

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
            if self.n == -1: # Amount of folds is not specified
                print("Error: CROSS_VALIDATOR -> amount of folds is not specified!")
                return None
            #return self.__divide_lists(X_data, y_data, k)
    
    def __divide_lists(X_data, y_data, ):
        for i in xrange(0, len(X_data), n):
            yield X_data[i:i + n], y_data[i:i + n]
        
        
        
cv = Cross_validator(1,1)
cv.cross_validate()
print ("Hello")
