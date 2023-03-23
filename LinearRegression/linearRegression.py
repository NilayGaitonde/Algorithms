import numpy as np
class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iters=100):
        print(learning_rate)
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights=None
        self.bias=None
        
    def fit(self,X,y):
        m,n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights)+self.bias
            dw = (1/m)*(np.sum(2*np.dot(X.T,(y_pred-y))))
            db = (1/m)*(np.sum(2*(y_pred-y)))
            print(dw,db)
            self.weights -= self.lr*dw
            self.bias -= self.bias*db
    
    def predict(self,X):
        return np.dot(X,self.weights)+self.bias