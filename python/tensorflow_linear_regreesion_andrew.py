from numpy import loadtxt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# loading data
input_matrix = loadtxt('input/ex1data1.txt', comments="#", delimiter=",", unpack=False)

x = input_matrix[:,0]
y = input_matrix[:,1]

x_as_matrix = np.reshape(x, (len(x), 1))
y_as_matrix = np.reshape(y, (len(y), 1))

x_as_tf_variable = tf.Variable(x_as_matrix, dtype=tf.float32)
y_as_tf_variable = tf.Variable(y_as_matrix, dtype=tf.float32)

W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1, 1]))

hypothesis = tf.matmul(x_as_tf_variable, W) + b
cost = 1 / (2*len(x)) * tf.reduce_sum(tf.square(hypothesis - y_as_tf_variable))

x_placeholder = tf.placeholder(tf.float32, [None, 1])
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

for i in range(0, 2000):
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1999]:
        fit = [sess.run(W)[0][0], sess.run(b)[0][0]]
        fit_fn = np.poly1d(fit)

        # tensorflow implementation

        plt.plot(x, y, 'yo', x, fit_fn(x), '--k')
        plt.xlim(0, 25)
        plt.ylim(0, 25)
        plt.show()

    sess.run(train_step, feed_dict={x_placeholder: x_as_matrix, y_placeholder: y_as_matrix})

print('W: {}'.format(sess.run(W)[0][0]))
print('b: {}'.format(sess.run(b)))
print('final cost: {}'.format(sess.run(cost)))

sess.close()
