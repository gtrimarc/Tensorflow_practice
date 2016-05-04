#############################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

#############################################################################
# TF constants
#
tf.InteractiveSession()

a = np.zeros((2,2))
ta = tf.zeros((2,2))
print a
#print ta

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b

with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())
    
#############################################################################
# TF variables and initialize_all_variables function
#
W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")

with tf.Session() as sess:
    print(sess.run(W1))
    sess.run(tf.initialize_all_variables())
    print(sess.run(W2))

#############################################################################
#
W = tf.Variable(tf.zeros((2,2)), name="weights")
R = tf.Variable(tf.random_normal((2,2)), name="random_weights")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(W))
    print(sess.run(R))

#############################################################################
# Updating variable state (example of a counter)
#
state = tf.Variable(0, name="counter")

# Define operations on TF variables
#
# Sum of a variable and a constant
new_value = tf.add(state, tf.constant(1))
# 
# "Equals" as an assign of a value from a variable to another variable
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print state.name
        print(sess.run(state))

print state.name
#############################################################################
# Fetching data from the graph
#
# Unless you update memory state there is no change in th graph??
#
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermed = tf.add(input2,input3)
mul = tf.mul(input1,intermed)

with tf.Session() as sess:
    result = sess.run([mul,intermed])
    print result

with tf.Session() as sess:
    result = sess.run([intermed,mul])
    print result
    
#############################################################################
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

#############################################################################
# Concept of placeholders and feed
#
# How about case of incompatible states? 
#
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    #
    # Idea of feed_dict : mapping from placeholders to concrete values
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))

#############################################################################
# Variable scope
#
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v",[1])
assert v.name == "foo/bar/v:0"

#############################################################################
# Weight sharing
#
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v",[1])
assert v1 == v
#
#############################################################################
# Understanding get_variable
#
#
#


#
#############################################################################
