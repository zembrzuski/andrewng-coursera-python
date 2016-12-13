
# this exercise is about neural network learning.


from numpy import loadtxt
import numpy as np
from xoxo import perform_gradient_checking_first_layer
from xoxo import perform_gradient_checking_second_layer
from xoxo import perform_gradient_checking_first_layer_regularization
from xoxo import perform_gradient_checking_second_layer_regularization

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


def compute_z(X, theta):
    return np.dot(X, theta.transpose())


def compute_a(z):
    return 1 / (1 + np.exp(-z))


# Now, I'll try to do feedforward propagation.

a1 = X
z2 = compute_z(a1, theta1)
a2 = compute_a(z2)
a2 = np.insert(a2, 0, 1, axis=1)
z3 = compute_z(a2, theta2)
a3 = compute_a(z3)
hypothesis = a3
print('a3 shape: {}'.format(a3.shape))

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
    """ g(z) """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """ g'(z) """
    sigm = my_sigmoid(z)
    # return np.dot(sigm, 1-sigm)
    return sigm * (1-sigm)


print("----")
z = 0.3
print(my_sigmoid(z))
print(sigmoid_gradient(z))

# I'll try gradient checking to be sure that my sigmoid gradient is correct.
epsilon = .001

checking = (my_sigmoid(z+epsilon)-my_sigmoid(z-epsilon))/(2*epsilon)
print(checking)
print(sigmoid_gradient(0))


# checking if my sigmoid gradient function performs well for matrix.

xoxo = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

checking = (my_sigmoid(xoxo+epsilon)-my_sigmoid(xoxo-epsilon))/(2*epsilon)
print(sigmoid_gradient(xoxo))
print(checking)


# Yep, my sigmoid_gradient function is correct.






#  2.3 Backpropagation
# ------------------------

# O professor pediu que cada iteração seja com somente um training example.
# Mas, para começar, vou fazer com all datasets.

# Step 1: feedforward propagation.

X = loadtxt('input/ex4data1.txt', comments="#", delimiter=",", unpack=False)

a1 = X
a1 = np.insert(a1, 0, 1, axis=1)
z2 = compute_z(a1, theta1)
a2 = compute_a(z2)
a2 = np.insert(a2, 0, 1, axis=1)
z3 = compute_z(a2, theta2)
a3 = compute_a(z3)
hypothesis = a3

# Step 2
error_layer_3 = (a3 - Y_matrix)         # (5000, 10)


# Step 3
my_dot = theta2.transpose().dot(error_layer_3.transpose()).transpose()
# Not sure if I need to remove this column here.
my_dot = np.delete(my_dot, 0, 1)
print(my_dot.shape)
print(sigmoid_gradient(z2).shape)
error_layer_2 = my_dot * sigmoid_gradient(z2)
print(error_layer_2.shape)


# Step 4

# Primeira coisa: vou dar um skip no esquema do delta.
# Primeiro, vou fazer as multiplicações para ver se o shape tá batendo.

# TODO completar meu delta layer
delta_layer_2 = error_layer_3.transpose().dot(a2)
delta_layer_1 = error_layer_2.transpose().dot(a1)


# Step 5
gradient_theta_layer_2 = delta_layer_2 / len(Y)
gradient_theta_layer_1 = delta_layer_1 / len(Y)

print("deu deu deu")

epsilon = .01


perceptron_index = 2
feature_index = 3

print("jajajajaja")


perform_gradient_checking_first_layer(
    epsilon, perceptron_index, feature_index, X, Y_matrix,
    theta1, theta2,
    gradient_theta_layer_1, gradient_theta_layer_2)

# Everything is performing fine now. I'll implement now regularized neural network.

# To achieve this goal, I need to change step 5, only.

llambda = 3

gradient_theta_layer_2_reg = 1 / len(Y) * delta_layer_2 + llambda / len(Y) * first_column_to_zeros(theta2)
gradient_theta_layer_1_reg = 1 / len(Y) * delta_layer_1 + llambda / len(Y) * first_column_to_zeros(theta1)

# Not sure if it is correct. I shall make gradient checking now.

print("ué, bombou")
perform_gradient_checking_first_layer_regularization(
    epsilon, perceptron_index, feature_index, X, Y_matrix,
    theta1, theta2,
    gradient_theta_layer_1_reg, gradient_theta_layer_2_reg, llambda)

print("testando a layer 2 agora")

perform_gradient_checking_second_layer_regularization(
    epsilon, perceptron_index, feature_index, X, Y_matrix,
    theta1, theta2,
    gradient_theta_layer_1_reg, gradient_theta_layer_2_reg, llambda)


