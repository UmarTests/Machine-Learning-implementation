import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import seaborn as sns

iris=load_iris()
df=sns.load_dataset('iris')
print(df.head)

# independent and dependent features
X=df.iloc[:,:-1] # independent features
print(X)
y=iris.target
print(y) # dependent features

# train the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)

# postpruning
treemodel=DecisionTreeClassifier(max_depth=3)
treemodel.fit(X_train,y_train)
DecisionTreeClassifier()

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)
plt.show()

y_pred=treemodel.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
