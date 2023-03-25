import pandas as pd
import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        m,n = X.shape

        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean,var and prior for each class

        self._mean = np.zeros((n_classes,n))
        self._var =np.zeros((n_classes,n))
        self._priors = np.zeros(n_classes)


        for index,c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[index,:] = X_c.mean(axis=0)
            self._var[index,:] = X_c.var(axis=0)
            self._priors[index] = X_c.shape[0]/float(n)

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        posteriors = []

        for index,c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            posterior = np.sum(np.log(self._pdf(index,x)))
            posterior = posterior+prior
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf(self,index,x):
        mean = self._mean[index]
        var = self._var[index]
        numerator = np.exp(-((x - mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)

        return numerator/denominator
