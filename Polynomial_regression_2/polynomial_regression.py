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
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

plt.scatter(X, y, color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 4)
X_polynomial = poly_features.fit_transform(X)
poly_features.fit(X_polynomial, y)

polynomial_regression = LinearRegression()
polynomial_regression.fit(X_polynomial, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, polynomial_regression.predict(poly_features.fit_transform(X)), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
linear_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
polynomial_regression.predict(poly_features.fit_transform(6.5))

""" 
    If you have any questions or suggestions regarding this script,
    feel free to contact me via nikhil.ss4795@gmail.com
"""
