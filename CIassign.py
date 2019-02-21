# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:07:40 2018

@author: Jameer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]
        errors_bias = (predictions - y) * X[:, 2]
        
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
        theta[2][0] = theta[2][0] - alpha * (1.0 / m) * errors_bias.sum()
        
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history




data=pd.read_csv('Desktop/CI - assign.csv',header=None)
"""
scatter(data[:, 0], data[:, 1], marker='o', c='blue')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
show()
"""

X1=data.iloc[0:1000,0:1]
X2=data.iloc[0:1000,1:2]


plt.scatter(X1,X2,color='red')

X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values

#number of training samples
m = Y.size
"""
#Add a column of ones to X (interception data)
it = ones(shape=(m, 2))
it[:, :1] = X
"""
#Initialize theta parameters
theta = zeros(shape=(3, 1))

#Some gradient descent settings
iterations = 1500
alpha = 0.01


print (compute_cost(X, Y, theta))
print (theta)
theta, J_history = gradient_descent(X, Y, theta, alpha, iterations)

print (theta)
print (compute_cost(X, Y, theta))

"""
result = X.dot(theta).flatten()
plot(data[:, 0], result)
"""
xcor = [-20,20] 
ycor = [-20,20]

plt.title("Intial  ") 
ycor[0] = (-1*theta[2]+(-1*theta[0]*xcor[0]))/theta[1]
ycor[1] = (-1*theta[2]+(-1*theta[0]*xcor[1]))/theta[1]
plt.scatter(X1,X2,color='red')
plt.plot(xcor,ycor)
plt.show() 

#Evaluate the linear regression
"""
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(X, Y, thetaT)

#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()
"""