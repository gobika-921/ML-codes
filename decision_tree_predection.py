from sklearn import datasets
iris = datasets.load_iris()

x= iris.data
y=iris.target

from sklearn import metrics
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
predection = clf.predict(x_test)

print(predection)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predection))