# 이건 진짜 필요한 경우가 아니면 안 돌려야지.
import tensorflow as tf
import numpy as np

# 그냥 input의 length를 가진 처음 layer를 input layer
# W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name="weight1")

# 마찬가지로 output의 length를 가진 마지막 layer를 output layer
# W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name="weight2")
# ...
# W10 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name="weight10")

# 중간에 unit 갯수를 지정하는 게 자유로는 layer가 hidden layer
# W11 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name="weight11")

# bias
# b1 = tf.Variable(tf.zeros([5]), name="bias1")
# ...
# b11 = tf.Variable(tf.zeros([1]), name="bias11")

# 연결, 연결
# with tf.name_scope("layer1") as scope:
#     L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# ...
# with tf.name_scope("layer10") as scope:
#     L10 = tf.nn.relu(tf.matmul(L9, W10) + b10)

# 마지막 sigmoid는 output과 연결되니까.
# with tf.name_scope("last") as scope:
#     hypothesis = tf.sigmoid(tf.matmul(L10, W11) + b11)
# output layer는 y의 범위를 [0, 1]로 제한하기 때문에 여전히 sigmoid를 사용한다.

# Vanishing gradient, 미분값이 chain rule에 의해 초기 layer로 올수록 weight가 작아져 output에 영향을 주지 못한다.
# 그래서 나온 것이 dumpping과 cut을 이용하는 random forest를 이용하거나 activation function을 ReLU로 바꾸는 것이다.


# 1. Activation function
# 마치 신경세포에서 역치 이상의 충격에 황성화되고, 그렇지 않으면 활성화되지 않은 것처럼 반응해서

# 순위표 (수렴도 + 보편적 사용빈도)
# 1. Leaky ReLU: max(0.1x, x)
# 2. ReLU: Rectified Linear Unit - 0, k < 0, k k >= 0 -> max(0, x)
# 3. tanh: tanh(x)
# 4. ELU: x, x > 0, alpha(exp(x) - 1), x <= 0
# 5. sigmoid: 1 / (1 + exp(-x))
# 6. Maxout: max(x*w1^t + b1, x*w2^t + b2)


# 2. initial value
# 0으로 초기화하는 건 weight를 측정할 수 없어 (계속 0) 그 영향을 체크할 수 없다.

# 2-1. RBM(Restricted Boatman Machine)'s Deep belief Net
# forward(일반 적합, encoding or decoding)를 통해 조정한 weight으로 backward(y를 통한 x 예측)을 실시한다.
# 이때 당연하게도 x에 대한 오차합이 작으면 작을수록 적절한 weight 값으로 판단한다.
# 이는 weight을 조정하고자 하는 이웃한 layer(2개)만을 가지고 하며, 이것을 각각의 weight 행렬마다 적용하면서 초기값을 지정한다.

# 2-2. Xavier[샤이벌?] initialization
# layer의 x(fan_in)와 y(fan_out)의 비례하는 값으로 초기화하자.
# W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
# 이보다 더 정밀한 초기값으로 (왜 2로 나누면 잘되는지 이유는 잘 모름)
# W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)


# avoid Over=fitted
# Regularization, shrinking parameter
# ohters: reduce the number of features, more training data

# addition) Dropout
# 마치 regularization의 최종이 의미없이 작은 계수를 가진 변수를 모델에서 제외시키듯 (결국엔 모수 절약으로 간다는)
# layer에서 weights 또는 unit을 일부(대체로 random) 지우자.
# Dropout으로 일부를 가지고 weight을 구성하고, 결과적으로 모형을 평가 및 이용할 때는 Dropout을 통해 구해진 모든 weight을 이용한다.

# addition) ensemble
# GAM처럼 얘기하셨는데, Dropout의 끝이나 random forest처럼 알고 있는데.
# 적게는 2%에서 크게는 4% 정도까지 정확도가 올라간다고 합니다.

# addition
# Fast forward: Drop-out이 모델 학습 때 몇 개의 input을 없애고(줄이고) 하는 거라면, Fast forward는 몇 개의 hidden layer를 건너 뛰어 학습시키는 것.
# 근데 함수 식을 보니까 몇 칸 뒤에 layer 구성에 input을 그냥 더하는 것 같기도?
# split를 통해 부분, 부분 모델 구성하는 split 관련에서 연속을 위해 원 변수를 더하는 것 같은 모양새.
# Split & merge: hidden unit을 통해 input layer의 수보다 hidden unit의 수가 적어졌다가 많아졌다가.
# Recurrent network, RNN: 앞으로 나가는 것에서 나아가 같은 layer에 다른 unit으로 가기도 하고 다시 나아가기도 하고.


# 실전, MNIST on NN
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

learning_rate = 0.001

# W1 = tf.Variable(tf.random_normal([784, 256]))
W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([256, 256]))
W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# W3 = tf.Variable(tf.random_normal([256, 10]))
W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print("Epoch: %04d" % (epoch + 1), "cost = {:.9f}".format(avg_cost))

    print("Accuracy:", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    import matplotlib.pyplot as plt
    import random

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1], keep_prob: 1}))
    # plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest")
    # plt.show()


# Optimizer
# Adam을 추천하신 답디다.
# Sub check: http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html
# tf.train.AdadeltaOptimizer
# tf.train.AdagradOptimizer
# tf.train.AdagradDAOptiimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOPtimizer
