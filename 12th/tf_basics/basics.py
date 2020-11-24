import tensorflow as tf

print("해당 module은 input을 MNIST data로 제한하고 있습니다.")


class Model:
    def __init__(self, sess, name, init_point=0):
        self.sess = sess
        self.name = name
        if init_point is 0:
            self.set_learning_rate()
            self._build_net()

    def set_learning_rate(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)

            # input placeholders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # First hidden layer
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            # L1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            # L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2, 2], padding="SAME", strides=2)
            # L1 = tf.layers.dropout(inputs=L1, rate=self.keep_prob, training=True)
            # return n, 14, 14, 32

            # Second hidden layer
            # 32x14x14의 그림을 관통하는 weights mask가 64개?
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            # return n, 7, 7, 64

            # Third hidden layer
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            # return n, 4, 4, 128
            L3 = tf.reshape(L3, [-1, 4 * 4 * 128])

            # Forth first Fully Connected (FC, Dense) layer
            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))

            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            # L4 = tf.layers.dense(inputs=L3, units=625, activation=tf.nn.relu)
            # L4 = tf.layers.dropout(inputs=L1, rate=self.keep_prob, training=True)
            # return 625, 625

            # Fifth second FC layer
            W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.hypothesis = tf.matmul(L4, W5) + b5

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})
