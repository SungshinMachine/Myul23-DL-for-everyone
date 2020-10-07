import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5],
          [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]
# class를 입력포맷에 맞게 바꿔서 3-class 전부에 대해 서술되어 있다.

# about model
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(cost)

# 보통 inline으로 구현 및 실행시키는 건 변수 복사? 안 해서 값 변경되지 않나.
# with는 기본 함수형으로 만든 건가.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # arg_max 맛보기
    total = sess.run(
        hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(total, sess.run(tf.argmax(total, 1)), sep="\n")
# argmax나 arg_max나 잘 되는 것 같은데, argmax가 2의 함수 이름과 같아서 별 다른 warning을 생성하지 않는다.

# Fancy Softmax Classifier
xy = np.loadtxt("data-o4-zoo.csv", delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# y_data = [[3], [3], [3], [2], [2], [2], [1], [1]]
nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
# 일반적으로 max값을 1로 하는 각각의 벡터를 반환하면서 중간 차원이 하나 추가되어 나온다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# -1: 정확한 갯수 지정을 피한다.

# W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
# b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={
                                     X: x_data,
                                     Y: y_data
                                 })
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # prediction이긴 한데, 모형 적합에 사용한 걸 써서 의미 없지 않나.
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        # for p, y in zip(pred, y_data):
        print(f"[{p == int(y)}] Prediction: {p} True Y: {int(y)}")
        # print(f"[{p == y}] Prediction: {p} True Y: {y}")
# flatten(): 개인적으론 차원 낮추기로 보이나, 열벡터를 행벡터로 바꾸는 데 사용되는 것 같다.
