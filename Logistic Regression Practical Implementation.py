import seaborn as sns
import pandas as pd
import numpy as np

# load iris dataset
from sklearn.datasets import load_iris
df=sns.load_dataset("iris")
# print(df.head())

# print(df['species'].unique()) # print unique values in species column

# to find missing values
# print(df.isnull().sum())

# to remove any category
df=df[df['species']!='setosa']
print(df.head())

df['species']=df['species'].map({'versicolor':0, 'virginica':1}) 
# map the species column to 0 and 1

print(df.head())

# splitting the data into independent and dependent features
X=df.iloc[:,:-1] #independent features
Y=df.iloc[:,-1] # dependent feature

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.model_selection import GridSearchCV
# hyperparameter tuning
parameter={'penalty':['l1','l2','elasticnet'], 'C':[1,.3,5,7,10,15,24,40,50], 'max_iter':[50,100,200,300]}
classifier_regressor=GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)
classifier_regressor.fit(X_train, Y_train)

print("Best Hyperparameters:", classifier_regressor.best_params_)
print("Best Cross-validation Score:", classifier_regressor.best_score_)
y_pred=classifier_regressor.predict(X_test)
print("Predictions:", y_pred)
# accuracy
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(Y_test, y_pred))

print("Classification Report:\n", classification_report(Y_test, y_pred))

# EDA
# plot pairplot
import matplotlib.pyplot as plt
sns.pairplot(df, hue='species')
plt.show()

print(df.corr())