"""
Created on Sat Sep 25 00:00:00 2018

@author: Nikhil
"""

""" 
    If you have any questions or suggestions regarding this script,
    feel free to contact me via nikhil.ss4795@gmail.com
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
soda_dataset = pd.read_csv('soda.csv')
X = soda_dataset.iloc[:, 0:1].values
y = soda_dataset.iloc[:, 1].values

plt.scatter(X, y, color = 'red')
plt.title('Temperature vs soda sales')
plt.xlabel('Temperature')
plt.ylabel('Sales (In units)')

soda_dataset.head()

soda_dataset.describe()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)

plt.scatter(X, y, color = 'red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Temperature vs Sales')
plt.xlabel('Temperature')
plt.ylabel('Sales (In units) ')
plt.show()

from sklearn import metrics
from sklearn.metrics import r2_score
print('Explained Variance Score :', metrics.explained_variance_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r^2 statistic: %.2f' % r2_score(y_test, y_pred))

## Accuracy : 88 (Using Linear Regression)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_train_poly_reg = poly_reg.fit_transform(X_train)
X_test_poly_reg = poly_reg.fit_transform(X_test)

polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_train_poly_reg, y_train)
y_pred = polynomial_regressor.predict(X_test_poly_reg)

# Visualising the Polynomial Regression results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(sorted(X_test), sorted(y_pred),color = 'blue')
plt.title('Temperature vs soda sales (Test set)')
plt.xlabel('temperature')
plt.ylabel('soda units sold')
plt.show()

from sklearn import metrics
from sklearn.metrics import r2_score
print('Explained Variance Score :', metrics.explained_variance_score(y_test, y_pred))
print('Mean Absolute Error :', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error :', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r^2 statistic : %.2f' % r2_score(y_test, y_pred))


## Accuracy : 95 (Using polynomial Regression)

""" 
    If you have any questions or suggestions regarding this script,
    feel free to contact me via nikhil.ss4795@gmail.com
"""
