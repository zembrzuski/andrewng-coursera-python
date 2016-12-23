from numpy import loadtxt
import tensorflow as tf
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
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Don't forget initializing the variables!
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
hyp = sess.run(hypothesis)
initial_cost = sess.run(cost)
print('initial cost: {}'.format(initial_cost))
print('---')

for i in range(1, 2000):
    sess.run(train_step, feed_dict={x_placeholder: x_as_matrix, y_placeholder: y_as_matrix})
    print('iter {} - cost: {}'.format(i, sess.run(cost)))

print('W: {}'.format(sess.run(W)))
print('b: {}'.format(sess.run(b)))
print('final cost: {}'.format(sess.run(cost)))

sess.close()
