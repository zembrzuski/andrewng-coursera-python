from numpy import loadtxt
import tensorflow as tf
import numpy as np

input_matrix = loadtxt('input/ex1data2.txt', comments="#", delimiter=",", unpack=False)

X = input_matrix[:,[0, 1]]
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

Y = input_matrix[:,-1]
Y_mean = Y.mean()
Y_std = Y.std()
Y_normalized = (Y - Y_mean) / Y_std

# Y must be a column vector.
Y_normalized = np.reshape(Y_normalized, (len(Y_normalized), 1))

x_as_tf_variable = tf.Variable(X_normalized, dtype=tf.float32)
y_as_tf_variable = tf.Variable(Y_normalized, dtype=tf.float32)

# maybe it is wrong :(
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))

hypothesis = tf.matmul(x_as_tf_variable, W) + b
cost = 1 / (2*len(X)) * tf.reduce_sum(tf.square(hypothesis - y_as_tf_variable))

x_placeholder = tf.placeholder(tf.float32, [None, 2])
y_placeholder = tf.placeholder(tf.float32, [None, 1])

train_step = tf.train.GradientDescentOptimizer(.01).minimize(cost)


# Don't forget initializing the variables!
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
hyp = sess.run(hypothesis)
initial_cost = sess.run(cost)
print('initial cost: {}'.format(initial_cost))
print('---')

for i in range(0, 500):
    print(sess.run(cost))
    sess.run(train_step, feed_dict={x_placeholder: X_normalized, y_placeholder: Y_normalized})

print('W: {}'.format(sess.run(W)))
print('b: {}'.format(sess.run(b)))
print('final cost: {}'.format(sess.run(cost)))

sess.close()
