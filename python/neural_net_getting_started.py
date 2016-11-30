import numpy as np

# I'll start implementing a feedward propagation.

# this should be an AND function

theta = np.array([-30, 20, 20])

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# adding bias therm.
print(X)
X = np.insert(X, 0, 1, axis=1)
print(X)


# feedforward propagation on a very simple neural net like this is
# absolutely equal to logistic regression.


def compute_hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))


print(compute_hypothesis(X, theta) > .5)

# It is working !



# Now, let us calculate an OR function.
# It is the same thing, with different theta
theta = np.array([-10, 20, 20])
print(compute_hypothesis(X, theta) > .5)

# Now, le us calculate NOT function.

X = np.array([
    [0],
    [1],
])

# adding bias therm.
X = np.insert(X, 0, 1, axis=1)

theta = np.array([10, -20])
print(compute_hypothesis(X, theta) > .5)

# So far, so easy. Now, we can compose these very simple functions
# to create a more sophisticated solution, like
# x1 XNOR x2

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# adding bias therm.
X = np.insert(X, 0, 1, axis=1)

x1ANDx2_theta = np.array([-30, 20, 20])
NOTx1_and_NOTx2_theta = np.array([10, -20, -20])
x1ORx2_theta = np.array([-10, 20, 20])

print("==========")


def createA(theta):
    def compute(X):
        z = np.dot(X, theta)
        return 1 / (1 + np.exp(-z))

    return compute


a1_2 = createA(x1ANDx2_theta)
a2_2 = createA(NOTx1_and_NOTx2_theta)
a1_3 = createA(x1ORx2_theta)

output_a1_2 = a1_2(X)
output_a2_2 = a2_2(X)

input_a2 = np.array([
    np.repeat(1, len(output_a1_2)),
    output_a1_2,
    output_a2_2
]).transpose()

output_a1_3 = a1_3(input_a2)

print(output_a1_3 > .5)
