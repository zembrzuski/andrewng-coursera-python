
# this exercise is about neural network learning.

from numpy import loadtxt
import numpy as np

X = loadtxt('input/ex4data1.txt', comments="#", delimiter=",", unpack=False)

# adding bias therm
X = np.insert(X, 0, 1, axis=1)

Y = loadtxt('input/ex4yy.txt', comments="#", delimiter=",", unpack=False)

theta1 = loadtxt('input/ex4theta1.txt', comments="#", delimiter=",", unpack=False)
theta2 = loadtxt('input/ex4theta2.txt', comments="#", delimiter=",", unpack=False)

print('x shape: {}'. format(X.shape))
print('y shape: {}'.format(Y.shape))

print('theta1 shape: {}'.format(theta1.shape))
print('theta2 shape: {}'.format(theta2.shape))

# In my thetas, each row means a perceptron.
# In my thetas, each column means a feature.


def compute_hypothesis(X, theta):
    z = np.dot(X, theta.transpose())
    return 1 / (1 + np.exp(-z))


# Now, I'll try to do feedforward propagation.

a1 = X
a2 = compute_hypothesis(a1, theta1)
a2 = np.insert(a2, 0, 1, axis=1)
a3 = compute_hypothesis(a2, theta2)
hypothesis = a3
print('a3 shape: {}'.format(a3.shape))
print(a3[1] > 0.5)
# It seems to be working. I'll implement the cost function now.


# I need to create a matrix with solutions before implementing the cost function now.

Y_matrix = np.zeros((len(Y), 10))
print(Y_matrix.shape)

for i in range(0, 5000):
    answer = int(Y[i]) - 1
    if answer == -1:
        answer = 9
    Y_matrix[i][answer] = 1



print('{} - {}'.format(hypothesis[0] > .5, Y_matrix[0]))
print('{} - {}'.format(hypothesis[500] > .5, Y_matrix[500]))
print('{} - {}'.format(hypothesis[4999] > .5, Y_matrix[4999]))

# Now, I think my Y matrix is correct!

# I am pretty sure my compute_cost function, implemented in the logistic regression
# exercise will work fine.

def compute_cost(hypothesis, Y):
    first_summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    return 1 / len(Y) * first_summation


print(compute_cost(hypothesis, Y_matrix))

# Yep! It is working, because the expected value is 0.287 too!

# Now, I'll implement cost_function with regularization. I need to make some changes
# on the cost function implemented on the logistic regression exercise, because
# there are two layers now. But the final solution will be pretty close from the original
# one.


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


print(compute_cost_with_regularization(hypothesis, Y_matrix, 1, theta1, theta2))

# It is working. If I wanted, I could do the compute_cost_function in a more generic way.
# I could process a n-layered neural network instead of a 2-layered neural netwok, but
# I am happy with this solution by now.



# Now, I need to compute the backpropagation: the most difficult part
# of this machine learning series.

# 2.1 - Sigmoid gradient.


def my_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.dot(my_sigmoid(z), 1-my_sigmoid(z))


print("----")
z = 0.3
print(my_sigmoid(z))
print(sigmoid_gradient(z))

# I'll try gradient checking to be sure that my sigmoid gradient is correct.
epsilon = .001

checking = (my_sigmoid(z+epsilon)-my_sigmoid(z-epsilon))/(2*epsilon)
print(checking)
print(sigmoid_gradient(0))

# Yep, my sigmoid_gradient function is correct.


print("backpropagation")

# 2.3 Backpropagation

# There are 4 steps

# 1st step: perform forward propagation (I have already done it. I'll just
# copy and paste it to organize the ideas.

theta1_random = np.random.random(theta1.shape)
theta2_random = np.random.random(theta2.shape)

a1 = X
a2 = compute_hypothesis(a1, theta1_random)
a2 = np.insert(a2, 0, 1, axis=1)
a3 = compute_hypothesis(a2, theta2_random)
hypothesis = a3

# 2nd step:
error_layer_3 = a3 - Y_matrix

# 3rd step:
# error_layer_2 = TODO
print('theta_2_shape: {}'.format(theta2_random.shape))
print('error_layer_3_shape: {}'.format(error_layer_3.shape))
print('----')



# aprendi as coisas
# organizacao dos times (pontos)
# design: escalar o sistema (minha maior contribuição: levar adiante integração com filas, elastic-search)
