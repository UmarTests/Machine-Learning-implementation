from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.1, 3.9, 5.2])

# Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

# Model
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_scaled, y_scaled)

# Predict
y_pred = sc_y.inverse_transform(svr.predict(X_scaled).reshape(-1,1))

print(y_pred)
import matplotlib.pyplot as plt
# visualization of SVR results
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Support Vector Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()