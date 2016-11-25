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

theta = np.array([float(0), float(0)])


def compute_hypothesis(X, theta):
    return np.dot(X, theta)


def compute_cost(hypothesis, y):
    return 1/(2*len(y)) * np.sum(np.square(hypothesis - y))


def compute_gradient(X, Y, hypothesis):
    return 1/(len(X)) * (X.transpose().dot(hypothesis - Y))


hypothesis = compute_hypothesis(X, theta)
cost = compute_cost(hypothesis, y)
print('the cost is {}'.format(cost))

# I think this is the gradient of the cost function relative to theta.
gradient = compute_gradient(X, y, hypothesis)
print('the gradient is {}'.format(gradient))



#
# let us do gradient checking to be sure that the compute_gradient function is correct.
#


def perform_gradient_checking(epsilon, theta, thetaIndex, X, Y):
    thetaA = np.array(theta)
    thetaB = np.array(theta)
    thetaA[thetaIndex] += epsilon
    thetaB[thetaIndex] -= epsilon
    hypothesisA = compute_hypothesis(X, thetaA)
    hypothesisB = compute_hypothesis(X, thetaB)
    checking = (compute_cost(hypothesisA, Y) - compute_cost(hypothesisB, Y)) / (2 * epsilon)
    gradient = compute_gradient(X, Y, compute_hypothesis(X, theta))
    print('checking  theta[{}] gradient: {} ~ {}'.format(thetaIndex, checking, gradient[thetaIndex]))


# first thing: checking if theta[0] gradient is ok:
perform_gradient_checking(.1, theta, 0, X, y)

# second thing: checking if theta[1] gradient is ok:
perform_gradient_checking(.1, theta, 1, X, y)

# Now I am sure that compute_gradient is performing fine! Let us do gradient descent now!

good_theta = np.array([float(1), float(1)])
alfa = 0.01

def perform_gradient_descent(num_iterations, X, Y, theta, alfa):
    good_theta = np.array(theta)
    for i in range(1, num_iterations):
        hypothesis = compute_hypothesis(X, good_theta)
        cost = compute_cost(hypothesis, Y)
        print(cost)
        good_theta = good_theta - alfa * compute_gradient(X, Y, hypothesis)
    return good_theta



print(perform_gradient_descent(2000, X, y, good_theta, 0.01))


###
###
###
### Now, I'll try to perform linear regression with multiple variables.
### I think the implementation that I have made is ready to perform it.
###
###
###


# 3.1 feature normalization

# reading second dataset

input_matrix = loadtxt('input/ex1data2.txt', comments="#", delimiter=",", unpack=False)

X = input_matrix[:,[0, 1]]
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std


# printing non-normalized and normalized datasets, to see if they seem similar.
#x = X[:,0]
#y = X[:,1]
#plt.scatter(x, y, alpha=0.5)
#plt.show()

#x = X_normalized[:,0]
#y = X_normalized[:,1]
#plt.scatter(x, y, alpha=0.5)
#plt.show()


Y = input_matrix[:,-1]
Y_mean = Y.mean()
Y_std = Y.std()
Y_normalized = (Y - Y_mean) / Y_std


# adding bias therm to x matrix
X_normalized = np.insert(X_normalized, 0, 1, axis=1)

theta=np.array([float(1), float(1), float(1)])

hypothesis = compute_hypothesis(X_normalized, theta)
compute_gradient(X_normalized, Y_normalized, hypothesis)


# now, I'll perform gradient checking again to be sure that compute_gradient
# is working fine. I am pretty sure it is working fine, because I hadn't problems
# calling the compute_gradient yet, but I prefer being so prudent now.

perform_gradient_checking(.1, theta, 0, X_normalized, Y_normalized)
perform_gradient_checking(.1, theta, 1, X_normalized, Y_normalized)
perform_gradient_checking(.1, theta, 2, X_normalized, Y_normalized)

# yep! now I am really sure that compute_gradient is working fine!
# I am ready to perform gradient descent with multiple variables!

perform_gradient_descent(100, X_normalized, Y_normalized, theta, .001)

# because of the cost is always decreasing, gradient descent is fine!
# I am done! I have finished my job by now :)
# thanks!
