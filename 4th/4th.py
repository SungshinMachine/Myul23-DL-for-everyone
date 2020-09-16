import tensorflow as tf
import matplotlib.pyplot as plt

print("tensorflow", tf.__version__, "ver. on")

# data set
X = [1, 2, 3]
Y = [1, 2, 3]

# model set
W = tf.placeholder(tf.float32)
hypothesis = X * W
# X_0를 1로 설정한 것이 아니라 b(y-절편)을 0으로 하는 원점으로의 선형회귀를 채택한 것이다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## plot용 값 생성
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# plotting
plt.plot(W_val, cost_val)
plt.show()
## 최근 trend가 axes를 이용할 때는 show() 멤버함수보단 세미콜론을 이용한 plot setting에 끝을 알려주는 것인 듯하다.
## "=="과 "is"처럼 후자가 더 빠른 연산을 하기 때문일수도?


# Gradient Descent 직접 구현
## data set
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

## model set
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost)

# gradient 값의 조정
# gvs = optimizer.compute_gradient(cost)
# apply_gradients = optimizer.apply_gradients(gvs)
## 현재 gradient 값을 받아 변경하고 다시 밀어넣는다. 현재는 아무것도 하지 않아 optimizer.minimize()한 것과 같다.

## Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(
        step,
        "cost:",
        sess.run(cost, feed_dict={X: x_data, Y: y_data}),
        sess.run(W),
        sep="\t",
    )

