from numpy import loadtxt
import matplotlib.pyplot as plt
import numpy as np

input_matrix = loadtxt('input/ex2data1.txt', comments="#", delimiter=",", unpack=False)

X = input_matrix[:, [0, 1]]
Y = input_matrix[:, -1]

# plotting scatterplot
# ----
# plt.scatter(X[:,0], X[:,1], alpha=0.6, c=Y)
# plt.show()

# adding bias therm to x matrix
X = np.insert(X, 0, 1, axis=1)
theta = np.array([float(1), float(1), float(1)])


def compute_hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))


def compute_cost(hypothesis, Y):
    return 1 / (2 * len(Y)) * np.sum(np.square(hypothesis - Y))


hypothesis = compute_hypothesis(X, theta)
print(compute_cost(hypothesis, Y))


y = [1, 0, 1]
h0 = [0.8, 0.4, 0.6]