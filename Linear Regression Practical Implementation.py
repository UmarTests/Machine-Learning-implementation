import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

df=fetch_california_housing()
print(pd.DataFrame(df.data))

dataset=pd.DataFrame(df.data)

# print column names
dataset.columns=df.feature_names
print(dataset.head())

# Independent and dependent features
X=dataset # Independent features
y=df.target # Dependent feature
print(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30  , random_state=42)  

# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
print(X_train[:5])  # Display first 5 rows of the standardized training data

X_test= scaler.transform(X_test)
print(X_test[:5])  # Display first 5 rows of the standardized testing data

#  to get back the original data
# X_train_original = scaler.inverse_transform(X_train)

from sklearn.linear_model import LinearRegression
# cross validation
from sklearn.model_selection import cross_val_score

regresion= LinearRegression()
regresion.fit(X_train, y_train)
mse=cross_val_score(regresion, X_train, y_train, cv=6, scoring='neg_mean_squared_error')

# print the cross validation score in array format
print("Cross-validation scores (negative MSE):", mse)

print(np.mean(mse))  # Mean of the cross-validation scores

# preddict
reg_pred=regresion.predict(X_test)
# print the first 5 predictions
print("First 5 predictions:", reg_pred[:5])

import seaborn as sns
sns.displot(reg_pred-y_test, kind="kde")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Distribution of Actual vs Predicted Values")
plt.show()

from sklearn.metrics import  r2_score
score2=r2_score( y_test,reg_pred)
print("R^2 Score:", score2)

plt.scatter(y_test, reg_pred, alpha=0.3)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
