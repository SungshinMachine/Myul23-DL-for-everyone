#

# XOR
# one hidden layer having two hidden units

# sigmoid가 중간 값을 처리하는 함수 정도라는 걸 자꾸 까먹는다. 4학년 수업은 다 잊어버렸나보다.
# 미분, 편미분, chain rule과 Back-propagation
# Back부터 미분을 통해 출력-layer에 대한 입력-layer의 각각의 순간변화율, 즉 변화 정도에 대하여 구할 수 있다.


import numpy as np
import tensorflow as tf

# data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

num_units = 10  # 2
# first hidden layer
W1 = tf.Variable(tf.random_normal([2, num_units]), name="weight1")
b1 = tf.Variable(tf.random_normal([num_units]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# second hidden layer
W2 = tf.Variable(tf.random_normal([num_units, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# 2개를 wide하게 쌓은 방법대로 더 많은 hidden layer를 만든 것을 deep하다고 하고, 이를 보편화시켜 Deep-Learning이라 한다.

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# 일일이 숫자 확인 -> TensorBoard를 이용한 확인
# 1. From TF graph, decide wihch tensors you want to log
w2_hist = tf.summary.histogram("weight2", W2)
cost_sum = tf.summary.scalar("cost", cost)

# 1-1. 만약 좌표축에 두 축 다 scalar가 아니라면
# w2_hist = tf.summary.histogram("weight2", W2)
# b2_hist = tf.summary.histogram("bias2", b2)
# hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# 1-2. 더 확장하자면, graph hierarchy, 앞선 선언도 합쳐서지 않을까?
# with tf.name_scope("layer1") as scope:
#     W1 = tf.Variable(tf.random_normal([2, num_units]), name="weight1")
#     b1 = tf.Variable(tf.random_normal([num_units]), name="bias1")
#     layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

#     w1_hist = tf.summary.histogram("weight1", W1)
#     b1_hist = tf.summary.histogram("bias1", b1)
#     layer1_hist = tf.summary.histogram("layer1", layer1)

# with tf.name_scope("layer2") as scope:
#     W2 = tf.Variable(tf.random_normal([num_units, 1]), name="weight2")
#     b2 = tf.Variable(tf.random_normal([1]), name="bias2")
#     hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#     w2_hist = tf.summary.histogram("weight2", W2)
#     b2_hist = tf.summary.histogram("bias2", b2)
#     hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# 2. Mer all summaries
summary = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 3. Create writer and add graph
    writer = tf.summary.FileWriter("C:/dump/logs")  # address
    writer.add_graph(sess.graph)

    for step in range(10001):
        # sess.run(train, feed_dict={X: x_data, Y: y_data})

        # 4. Run summary mer and add_summary
        s, _ = sess.run([summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s, global_step=step)

        if step % 100 == 0:
            print(
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                sess.run([W1, W2]),
            )

    # 사실 적합시킨 데이터를 준 꼴이라 정말 높게 안 나오면 그냥 망.
    h, c, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )
    print(f"Hypothesis: {h}", f"Correct: {c}", f"Accuracy: {a}", sep="\n")

# 5. Launch TensorBoard
# in terminal, 위와 동일한 address
# terminal code에 대해선 따로 공부가 필요할 듯
# tensorboard --logdir = C:\dump\logs

# 난 port num이 아니라 address를 주는데, 왜 안 되냐.
# 설마 local address 시작 port?가 달라서 그런가.

# 5-1. remote server
# ssh -L local_port:127.0.0.1:remote_port server_address
# tensorboard --logdir = C:\dump\logs
