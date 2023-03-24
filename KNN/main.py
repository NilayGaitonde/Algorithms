from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNearestNeighbors import KNN
import numpy as np


cmap = ListedColormap(['#FF0000' , '#00FF00', '#0000FF'])
iris=load_breast_cancer()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
plt.figure()
plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolors='k',s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

accuracy_1 = np.mean(predictions==y_test)*100

# round accuracy_1 to only show 2 digits after the decimal

print(f"Your accuracy:{round(accuracy_1,2)}%")