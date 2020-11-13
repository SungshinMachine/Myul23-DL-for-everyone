# Train-test / (CV) Train-Validation-test or LOOCV / (k-fold CV) repeat(Train[-i]-Validation[i])-test / bootstrap
# Online learning: repeat type CV 얘긴 거 같던데.

# Over-fitting: 학습 데이터에 너무 의존한 modeling, noise를 제거하지 않았으므로 model 해석이 상당히 까다롭다.
# 또, test data나 이후 얻게 된 data에 대해 training data만큼의 적합도, 설명력을 기대할 수 없다.
# 이를 제거하고자 More gathering training data, Reduce the number of features, Regularzation 등을 사용한다.
# 좀 더 효과적으로 overfitting을 제거하고자 the Lasso 방법도 사용하는데, 왜 아무도 언급하지 않지? tensorflow는 Lasso는 지원을 안 하나?

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1.
# learning rate - small: takes too long, stops a local optimum
#               - big: (over-shooting) will not converge, be divergency

# scikitlearn이었던가에 train-test 분할 함수가 있는데, default ratio는 train:test = 7:3이다.
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# learning_rate을 크게 주면, 당연히 발산하면서 cost는 Inf로 W-matrix는 NaN으로

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, "\n", W_val)

    print("Prediction:", sess.run(prediction, feed_dict={X: x_data}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

# 2.
# cost에 대한 contour가 굉장히 찌그러진 타원을 형성할 때는 alpha값이 충분히 작아도 global optimum을 찾지 못하고 발산할 수 있다.
# 이럴 때는 contour의 전체적 공간축의 범위를 비슷하게 조정해줄 필요가 있다. -> normalization
# 또, 직관적인 이해 등을 이유로 zero-centering을 한다.
# normalization + zero-centering => standardization의 일종

xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)
xy = MinMaxScaler().fit_transform(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        print(step, "Cost:", cost_val, "\nPrediction\n", hy_val)

# 추가) MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# 오.. 정말 다운 받네요? 와..
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cross entrophy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        print("Epoch: %04d" % (epoch + 1), "cost = {:.9f}".format(avg_cost))

    print("Accuracy:", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # 그냥 시각적으로 표현하고 싶으셨던 듯?
    import matplotlib.pyplot as plt
    import random

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}))
    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest")
    plt.show()
