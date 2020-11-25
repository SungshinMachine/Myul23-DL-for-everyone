# 와, CNN, Convolutional Neural Network
# Stride: filter를 움직이는 칸의 수
# (matrix.cols - kernel.cols) / stride + 1 (중간 나눗셈에서 소수 나오는 거 처리 안해서 소수점 나오면 그냥 못하는 것.)

# 기본 정의에 따르면 filter or mask를 통해 (대체로 영상) 일부를 대표하는 값을 반환해 더 적은 수의 행렬로 구성한다.
# 이로 인한 데이터의 손실(가장자리 데이터는 덜 쓰임)을 막고자 주변을 의미없는 값(0)으로 두르는 작업(padding)을 한다.
# 이 떄문에 Stride만큼 padding을 두르면 input과 같은 크기의 matrix를 반환 받는다.
# Activation map: filter(s)를 통해 처리된 행렬

# Pooling (alike sampling)
# (영상 처리를 기준으로) input의 hyper-plane마다 대푯값 찾기를 진행해 부피가 작은 input을 구성하는 것.
# sampling이라고 했으니까 2x2 차원에서 줄이는 거라고 생각됨.
# 이때 mask 형태의 filter를 이용하는 방식이 아닌 filter의 대응 원소 중, 최댓값을 반환하는 방식을 MAX POOLING이라고 한다.

# Fully Connected Layer
# 찾아본 바에 의하면 output layer 앞 즈음에 flattn으로 나눠진 벡터들을 단일 벡터로 묶는 layer라는데,
# output의 volume에 맞춰 여러 input을 matrix 차원에서 재구성하는 단계로 보인다.
# 대체로 단일 벡터로 합치는 layer, 약간의 조정을 거치는 layer, 크기를 줄이는 layer 등 다중으로 구성되는 것 같다.

# Cases: AlexNet(2012), GoogLeNet(2014), ResNet(2015), AlphaGo


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 작은 데이터로 함수 알기 과정
image = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
weight = tf.constant([[[[1.0, 10.0, -1.0]], [[1.0, 10.0, -1.0]]], [[[1.0, 10.0, -1.0]], [[1.0, 10.0, -1.0]]]])

print(f"image.shape: {image.shape}, weight.shape: {weight.shape}")
plt.imshow(image.reshape(3, 3), cmap="Greys")
plt.show()  # 와 이제 알았는데, black formatter는 ;를 지워버리는구나.

sess = tf.InteractiveSession()
# input dimension에 대해 계속 생각할 것.
conv = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
# SAME: stride가 한 칸씩 움직인다면 output.dim이 input.dim과 같도록
conv_img = conv.eval()
print(f"conv_img.shape: {conv_img.shape}")

conv_img = np.swapaxes(conv_img, 0, 3)
for i, one_img in enumerate(conv_img):
    # padding = SAME으로 CNN 이후에 3x3의 행렬이 되었으므로
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1)
    plt.imshow(one_img, cmap="gray")
plt.show()


# mnist로 확인 과정
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Color image 아님, 다 Grayscale
img = mnist.train.images[0].reshape(28, 28)
img = img.reshape(-1, 28, 28, 1)
plt.imshow(img, cmap="gray")
plt.show()

rows, cols, color sheets, kernels
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))

sess = tf.InteractiveSession()
conv = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding="SAME")
# stride가 두 칸씩 움직인다면 아마도 input.rows/2, input.cols/2. input's dimension과 맞춰서 이런 모양?
sess.run(tf.global_variables_initializer())
conv_img = conv.eval()

conv_img = np.swapaxes(conv_img, 0, 3)
for i, one_img in enumerate(conv_img):
    plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(14, 14), cmap="gray")
plt.show()

pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()

pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i + 1)
    plt.imshow(one_img.reshape(7, 7), cmap="gray")
plt.show()


# Deep Learning으로의 모형 구축
learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)


X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])


# First hidden layer
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# return n, 14, 14, 32

# Second hidden layer
# 32x14x14의 그림을 관통하는 weights mask가 64개?
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
# return n, 7, 7, 64

# Third hidden layer
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
# return n, 4, 4, 128
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])


# Forth first Fully Connected (FC, Dense) layer
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))

L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
# return 625, 625

# Fifth second FC layer
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Learning started. It takes sometimes.")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print("Epoch: %4d" % (epoch + 1), "Cost: {:.9f}".format(avg_cost))

    print("Learning Finished!")
    print("Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
