from numpy import loadtxt
import numpy as np


def compute_hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))


input_matrix = loadtxt('input/ex2data2.txt', comments="#", delimiter=",", unpack=False)

X = input_matrix[:, [0, 1]]
Y = input_matrix[:, -1]
X = np.insert(X, 0, 1, axis=1)


def compute_cost_with_regularization(hypothesis, Y, llambda, theta):
    thetas_for_regularization = np.array(theta)
    thetas_for_regularization[0] = 0
    first_summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    second_summation = np.sum(np.square(thetas_for_regularization))
    return 1 / len(Y) * first_summation + llambda / (2 * len(Y)) * second_summation


def compute_gradient_regularization(X, Y, hypothesis, llambda, theta):
    thetas_for_regularization = np.array(theta)
    thetas_for_regularization[0] = 0
    return 1 / (len(X)) * (X.transpose().dot(hypothesis - Y)) + llambda / len(X) * thetas_for_regularization


def perform_gradient_descent_with_regularization(num_iterations, X, Y, theta, alfa, llambda):
    good_theta = np.array(theta)
    for i in range(1, num_iterations):
        hypothesis = compute_hypothesis(X, good_theta)
        cost = compute_cost_with_regularization(hypothesis, Y, llambda, theta)
        good_theta = good_theta - alfa * compute_gradient_regularization(X, Y, hypothesis, llambda, theta)
    return {'theta': good_theta, 'cost': cost}


print(perform_gradient_descent_with_regularization(200, X, Y, np.array([float(1), float(1), float(1)]), .1, 2))
