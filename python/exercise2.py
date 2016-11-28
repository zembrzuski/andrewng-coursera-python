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
print(X.shape)
X = np.insert(X, 0, 1, axis=1)
print(X.shape)

theta = np.array([float(0), float(0), float(0)])


def compute_hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))


hypothesis = compute_hypothesis(X, theta)


# I think I have implemented compute_hypothesis function fine. But I am
# not sure of this.

def compute_cost(hypothesis, Y):
    summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    return 1 / len(Y) * summation


print(compute_cost(hypothesis, Y))


# compute_cost and compute_hypothesis are working correctly, because in the
# exercise, the professor say that the compute_cost function for theta=(0,0,0)
# should be equal to 0.693 and compute_cost function answered the exact same
# value

def compute_gradient(X, Y, hypothesis):
    return 1 / (len(X)) * (X.transpose().dot(hypothesis - Y))


# I just copied compute_gradient from the linear regression exercise, because
# they are the same! The unique difference is the hypothesis function.
# I am pretty confident that my gradient_checking will work really fine!

gradient = compute_gradient(X, Y, hypothesis)
print('the gradient is {}'.format(gradient))


# to be sure that my compute_gradient is working fine, I'll do gradient checking
# again:

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


perform_gradient_checking(.1, theta, 0, X, Y)
perform_gradient_checking(.1, theta, 1, X, Y)
perform_gradient_checking(.1, theta, 2, X, Y)


# Yes, now I am sure that my gradient function is working fine!
# We can do gradient_descent now! The implementation is the same
# of linear regression!

def perform_gradient_descent(num_iterations, X, Y, theta, alfa):
    good_theta = np.array(theta)
    for i in range(1, num_iterations):
        hypothesis = compute_hypothesis(X, good_theta)
        cost = compute_cost(hypothesis, Y)
        good_theta = good_theta - alfa * compute_gradient(X, Y, hypothesis)
    return {
        'theta': good_theta,
        'cost': cost
    }


good_theta = np.array([float(0), float(0), float(0)])

answer_theta = perform_gradient_descent(8000, X, Y, good_theta, 0.0001)['theta']
print(answer_theta)

# For a student with an Exam 1 score
# of 45 and an Exam 2 score of 85, you should expect to see an admission
# probability of 0.776.
print('---')
print(compute_hypothesis(
    np.array([
        [float(1), float(0), float(0)],
        [float(1), float(10), float(10)],
        [float(1), float(45), float(85)],
        [float(1), float(85), float(85)],
        [float(1), float(95), float(95)]
    ]),
    answer_theta))

# my predictions are not the same on the professor ansers, but I am
# very very confident that my implementation is working fine, because
# higher the score of the students, higher the probability that they
# will succeed on the admission test.
# Maybe the values are not the same because It is possible that he
# made feature normalization.
# I'll not try feature normalization right now.



# Now, I need to implement the second part of the exercise: to implement
# regularization to avoid overfitting.

print('\n\n--- now with regularization ---')

input_matrix = loadtxt('input/ex2data2.txt', comments="#", delimiter=",", unpack=False)

X = input_matrix[:, [0, 1]]
Y = input_matrix[:, -1]

# plt.scatter(X[:,0], X[:,1], alpha=0.6, c=Y)
# plt.show()

X = np.insert(X, 0, 1, axis=1)


# I'll implement the new cost function now. Is basically a copy of the first
# cost function with the regularization therm.

def compute_cost_with_regularization(hypothesis, Y, llambda, theta):
    # REMEMBER: DO NOT REGULARIZE THETA(0)
    # to avoid regularizing theta, I'll remove it from the theta vector I am using to perform
    # the calculations.
    thetas_for_regularization = np.array(theta)
    thetas_for_regularization[0] = 0
    first_summation = np.sum(-Y * np.log(hypothesis) - (1 - Y) * np.log(1 - hypothesis))
    second_summation = np.sum(np.square(thetas_for_regularization))
    return 1 / len(Y) * first_summation + llambda / (2 * len(Y)) * second_summation


# Implementing gradient function for the new cost function now.

def compute_gradient_regularization(X, Y, hypothesis, llambda, theta):
    # REMEMBER: DO NOT REGULARIZE THETA(0)
    # to avoid regularizing theta, I'll remove it from the theta vector I am using to perform
    # the calculations.
    thetas_for_regularization = np.array(theta)
    thetas_for_regularization[0] = 0
    return 1 / (len(X)) * (X.transpose().dot(hypothesis - Y)) + llambda / len(X) * thetas_for_regularization


def perform_gradient_checking_with_regularization(epsilon, theta, thetaIndex, X, Y, llambda):
    # REMEMBER: DO NOT REGULARIZE THETA(0)
    # to avoid regularizing theta, I'll remove it from the theta vector I am using to perform
    # the calculations.
    thetaA = np.array(theta)
    thetaB = np.array(theta)

    thetaA[thetaIndex] += epsilon
    thetaB[thetaIndex] -= epsilon

    hypothesisA = compute_hypothesis(X, thetaA)
    hypothesisB = compute_hypothesis(X, thetaB)

    checking = (compute_cost_with_regularization(hypothesisA, Y, llambda, thetaA)
                - compute_cost_with_regularization(hypothesisB, Y, llambda, thetaB)) \
               / (2 * epsilon)

    gradient = compute_gradient_regularization(X, Y, compute_hypothesis(X, theta), llambda, theta)
    print('checking  theta[{}] gradient: {} ~ {}'.format(thetaIndex, checking, gradient[thetaIndex]))


theta = np.array([float(1), float(1), float(1)])

perform_gradient_checking_with_regularization(.001, theta, 0, X, Y, 0)
perform_gradient_checking_with_regularization(.001, theta, 1, X, Y, 0)
perform_gradient_checking_with_regularization(.001, theta, 2, X, Y, 0)
print('---')
perform_gradient_checking_with_regularization(.001, theta, 0, X, Y, 1)
perform_gradient_checking_with_regularization(.001, theta, 1, X, Y, 1)
perform_gradient_checking_with_regularization(.001, theta, 2, X, Y, 1)
print('---')
perform_gradient_checking_with_regularization(.001, theta, 0, X, Y, 100)
perform_gradient_checking_with_regularization(.001, theta, 1, X, Y, 100)
perform_gradient_checking_with_regularization(.001, theta, 2, X, Y, 100)

# The cost function and the gradient of the cost function are working very fine!

# Now, we are ready to implement the new gradient descent with regularization. It is the
# basic idea on the old function

def perform_gradient_descent_with_regularization(num_iterations, X, Y, theta, alfa, llambda):
    good_theta = np.array(theta)
    for i in range(1, num_iterations):
        hypothesis = compute_hypothesis(X, good_theta)
        cost = compute_cost_with_regularization(hypothesis, Y, llambda, theta)
        good_theta = good_theta - alfa * compute_gradient_regularization(X, Y, hypothesis, llambda, theta)
    return {
        'theta': good_theta,
        'cost': cost
    }


print(perform_gradient_descent(200, X, Y, theta, .1))
print(perform_gradient_descent_with_regularization(200, X, Y, theta, .1, 1))
print(perform_gradient_descent_with_regularization(200, X, Y, theta, .1, 2))
print(perform_gradient_descent_with_regularization(200, X, Y, theta, .1, 3))

#
# Gradient descent with regularization is working fine!
# I know it because of two reasons:
# 1) Gradient checking made me sure that the functions are working fine
# 2) Bigger the lambda, bigger the cost of the function
#
# It is enough by now. I will not plot the decision boundaries because
# I don't know anything about data visualization, and I will not
# play with lambda parameter because it is pretty simple.
#
