import numpy as np

def first_column_to_zeros(inp):
    return np.column_stack((np.zeros(len(inp)), np.delete(inp, 0, 1)))

def compute_cost_with_regularization(hypothesis, Y, llambda, theta_first_layer, theta_second_layer):
    thetas_1st = np.array(theta_first_layer)
    thetas_2nd = np.array(theta_second_layer)

    # do not regularize first column!
    thetas_1st = first_column_to_zeros(thetas_1st)
    thetas_2nd = first_column_to_zeros(thetas_2nd)

    first_summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    second_summation = np.sum(np.square(thetas_1st)) + np.sum(np.square(thetas_2nd))

    return 1 / len(Y) * first_summation + llambda / (2 * len(Y)) * second_summation

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







def perform_gradient_checking_first_layer_regularization(
        epsilon, perceptron_index, feature_index, X, Y,
        theta_layer_1, theta_layer_2,
        gradient_theta_layer_1_reg, gradient_theta_layer_2_reg, llambda):

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

    costA = compute_cost_with_regularization(hypothesisA, Y, llambda, thetaA, theta_layer_2)
    costB = compute_cost_with_regularization(hypothesisB, Y, llambda, thetaB, theta_layer_2)

    derivativeA = (costA - costB) / (2 * epsilon)
    derivativeB = gradient_theta_layer_1_reg[perceptron_index][feature_index]

    print('{} - {}'.format(derivativeA, derivativeB))




def perform_gradient_checking_second_layer_regularization(
        epsilon, perceptron_index, feature_index, X, Y,
        theta_layer_1, theta_layer_2,
        gradient_theta_layer_1_reg, gradient_theta_layer_2_reg, llambda):

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

    costA = compute_cost_with_regularization(hypothesisA, Y, llambda, theta_layer_1, thetaA)
    costB = compute_cost_with_regularization(hypothesisB, Y, llambda, theta_layer_1, thetaB)

    derivativeA = (costA - costB) / (2 * epsilon)
    derivativeB = gradient_theta_layer_2_reg[perceptron_index][feature_index]

    print('{} - {}'.format(derivativeA, derivativeB))
