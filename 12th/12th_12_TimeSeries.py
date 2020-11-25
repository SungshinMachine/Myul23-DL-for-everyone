import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


timesteps = seq_length = 7
input_dim = 5
learning_rate = 0.01
hidden_dim = 4
output_dim = 1

xy = np.loadtxt("data-02-stock_daily.csv", delimiter=",")
xy = xy[::-1]
xy = MinMaxScaler().fit_transform(xy)

x = xy
y = xy[:, [-1]]

x_data = []
y_data = []
for i in range(len(y) - seq_length):
    _x = x[i : i + seq_length]
    _y = y[i + seq_length]

    x_data.append(_x)
    y_data.append(_y)

train_size = 0.7
trainX, testX, trainY, testY = train_test_split(x, y, train_size=train_size)


X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])


cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# (from) many to one
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

loss = tf.reduce_mean(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print(f"{i}th, Loss: {l}")

    import matplotlib.pyplot as plt

    pred = sess.run(Y_pred, feed_dict={X: testX})
    plt.plot(testY, color="r", label="test")
    plt.plot(pred, color="b", label="prediction")
    plt.show()
