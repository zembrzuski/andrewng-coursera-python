import numpy as np


def compute_cost(hypothesis, Y):
    first_summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    return 1 / len(Y) * first_summation


def compute_z(X, theta):
    return np.dot(X, theta.transpose())


def compute_a(z):
    return 1 / (1 + np.exp(-z))


def perform_gradient_checking_first_layer(
        epsilon, perceptron_index, feature_index, X, Y,
        theta_layer_1, theta_layer_2,
        gradient_theta_layer_1, gradient_theta_layer_2):

    thetaA = np.array(theta_layer_1)
    thetaA[perceptron_index][feature_index] += epsilon

    thetaB = np.array(theta_layer_1)
    thetaB[perceptron_index][feature_index] -= epsilon

    a1 = X
    a1 = np.insert(a1, 0, 1, axis=1)
    z2 = compute_z(a1, thetaA)
    a2 = compute_a(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = compute_z(a2, theta_layer_2)
    a3 = compute_a(z3)
    hypothesisA = a3

    a1 = X
    a1 = np.insert(a1, 0, 1, axis=1)
    z2 = compute_z(a1, thetaB)
    a2 = compute_a(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = compute_z(a2, theta_layer_2)
    a3 = compute_a(z3)
    hypothesisB = a3

    derivativeA = (compute_cost(hypothesisA, Y) - compute_cost(hypothesisB, Y)) / (2 * epsilon)
    derivativeB = gradient_theta_layer_1[perceptron_index][feature_index]

    print('{} - {}'.format(derivativeA, derivativeB))





def perform_gradient_checking_second_layer(
        epsilon, perceptron_index, feature_index, X, Y,
        theta_layer_1, theta_layer_2,
        gradient_theta_layer_1, gradient_theta_layer_2):

    thetaA = np.array(theta_layer_2)
    thetaA[perceptron_index][feature_index] += epsilon

    thetaB = np.array(theta_layer_2)
    thetaB[perceptron_index][feature_index] -= epsilon

    a1 = X
    a1 = np.insert(a1, 0, 1, axis=1)
    z2 = compute_z(a1, theta_layer_1)
    a2 = compute_a(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = compute_z(a2, thetaA)
    a3 = compute_a(z3)
    hypothesisA = a3

    a1 = X
    a1 = np.insert(a1, 0, 1, axis=1)
    z2 = compute_z(a1, theta_layer_1)
    a2 = compute_a(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = compute_z(a2, thetaB)
    a3 = compute_a(z3)
    hypothesisB = a3

    derivativeA = (compute_cost(hypothesisA, Y) - compute_cost(hypothesisB, Y)) / (2 * epsilon)
    derivativeB = gradient_theta_layer_2[perceptron_index][feature_index]

    print('{} - {}'.format(derivativeA, derivativeB))


