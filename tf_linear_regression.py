#############################################################################
# Linear regression : setting up the data
#
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# import seaborn

# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# Plot input data
plt.scatter(X_data, y_data)
plt.show()
#
#############################################################################
# Linear regression in TensorFolow
#
tf.InteractiveSession()
n_samples = 1000
batch_size = 100

# Some reshaping of data
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))

# Define placeholders for input
X = tf.placeholder(tf.float32,shape=(batch_size,1))
y = tf.placeholder(tf.float32,shape=(batch_size,1))

# Define vairables to be learned
with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weights", (1,1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer())
    y_pred = tf.matmul(X,W) + b
    loss = tf.reduce_sum(((y-y_pred)**2)/n_samples)
    

opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(15000):
        indices = np.random.choice(n_samples,batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val = sess.run([opt_operation, loss],feed_dict={X: X_batch, y: y_batch})
        #print loss_val
print loss_val
#############################################################################
