import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris =load_iris()
test_index =[0,50,100]

train_data =np.delete(iris.data, test_index, axis=0)
train_target = np.delete(iris.target, test_index)

test_data = iris.data[test_index]
test_target = iris.target[test_index]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)


print("Prediction" ,clf.predict(test_data))