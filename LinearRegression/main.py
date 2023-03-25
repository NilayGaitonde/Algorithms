import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegression
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


X,y = make_regression(n_samples=700,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
print(f"{X_test.shape}\n{y_test.shape}\n{X_train.shape}\n{y_train.shape}")
print(f"{type(X_test)}\n{type(y_test)}\n{type(X_train)}\n{type(y_train)}")


# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],y,color="b",marker="o",s=30)
# plt.show()

regressor = LinearRegression(learning_rate=0.001,n_iters=1000)
regressor.fit(X_train,y_train)
predicted = regressor.predict(X_test)

mse_values = mse(y_test,predicted)
print(mse_values)


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,color = cmap(0.9), s=10)
m2 = plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color="black",linewidth=2,label="Prediction")
plt.show()