import numpy as np
import tensorflow as tf

def discr(var):
    constant = tf.constant(1.)
    return var + constant

def gen(var):
    return tf.reduce_sum(var)

x = tf.placeholder(tf.float32, [])
z = tf.placeholder(tf.float32, [2])

with tf.variable_scope("incrementation") as scope:
    D = discr(x)
with tf.variable_scope(scope, reuse=True):
    Dg = discr(gen(z))

xinput = 1.0
zinput = np.array([1., 1.])

lossD = D + Dg
lossG = Dg

with tf.Session() as sess:
    print(sess.run(lossD, feed_dict={x: xinput, z: zinput}))
    print(sess.run(lossG, feed_dict={z: zinput}))
