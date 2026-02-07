from sklearn import tree
features = [[150 , 1], [170, 1], [140 , 0], [130, 0], [190, 1], [120, 0]] # features are weight in grams and texture , 1 -> bumpy , 0 -> smooth 
labels = [1, 1, 0, 0, 1, 0]  # 1 -> orange, 0 -> apple
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print (clf.predict([[190, 1]]))  
