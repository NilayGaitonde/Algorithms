import pandas as pd
import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        predictions = [self.predict_single(x) for x in X]
        return predictions
    
    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def predict_single(self,x):
        euclidean_dist = [self.euclidean_distance(x,x_train) for x_train in self.X_train]
        k_neighbors_indices=np.argsort(euclidean_dist)[:self.k]
        k_neighbors_labels=[self.y_train[i] for i in k_neighbors_indices]
        print(self.X_train.shape)
        counter = Counter(k_neighbors_labels)
        return counter.most_common()[0][0]