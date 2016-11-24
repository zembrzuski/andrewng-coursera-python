from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

"""
This is my approach to solve linear regression
from machine learning course from professor
Andrew NG on Cousera.

Hope you enjoy.

Rodrigo Claro Zembrzuski
zembrzuski@gmail.com
"""


# loading data
input_matrix = loadtxt('input/ex1data1.txt', comments="#", delimiter=",", unpack=False)
x = input_matrix[:,0]
y = input_matrix[:,1]

# plotting scatterplot
# ---
# plt.scatter(x, y, alpha=0.5)
# plt.show()

# adding bias term to X matrix
ones_column = np.repeat(1, len(x))
X = np.transpose(np.array([ones_column, x]))

theta = [0, 0]


def compute_hypothesis(X, theta):
    return np.dot(X, theta)


def compute_cost(hypothesis, y):
    return 1/(2*len(y)) * np.sum(np.square(hypothesis - y))


def compute_gradient(X, Y, hypothesis):
    return 1/(len(X)) * (X.transpose().dot(hypothesis - Y))
    #return 1/(len(X)) * ((hypothesis - Y) @ X)


hypothesis = compute_hypothesis(X, theta)
cost = compute_cost(hypothesis, y)
print('cost is {}'.format(cost))

# I think this is the gradient of the cost function relative to theta.
gradient = compute_gradient(X, y, hypothesis)
print('gradient is {}'.format(gradient))

# gradient checking

first = np.array([1, 2, 3])

X = np.array([
    [1, 2],
    [3, 4],
    [1, 1]
])

print('aaa')
print(first.shape)
print(X.shape)
print(X.transpose().dot(first))

epsilon = 0.001
thetaPositive = np.array(theta)
thetaNegative = np.array(theta)
thetaPositive[0] += epsilon
thetaNegative[0] -= epsilon
hypothesisEpsilonPositive = compute_hypothesis(X, thetaPositive)
hypothesisEpsilonNegative = compute_hypothesis(X, thetaNegative)
gradient_aproximation = ((compute_cost(hypothesisEpsilonPositive, y) - compute_cost(hypothesisEpsilonNegative, y))) / (2*epsilon)
print('gradient approximation is {}'.format(gradient_aproximation))

"""
# Now, I'll try to do gradient checking.

y = 2x² + 3x + 1
y' = 4x + 3

como eu faço o gradient checking?
escolho um valor para x
x = 2

calculo y p/ 2+0.001
calculo y p/ 2-0.001

calculo o valor de y1 p/ 2
tem que ser iguais os valores

print(((2*2.001**2 + 3*2.001 + 1) - (2*1.999**2 + 3*1.999 + 1)) / (2*0.001))
"""

