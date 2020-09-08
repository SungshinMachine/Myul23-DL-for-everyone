import tensorflow as tf

# 그냥 해보고 싶어서.. tensorflow 버전 확인
if tf.__version__ <= "1.15.0":
    print("tensorflow 1.x ver. up")
else:
    print("tensorflow 2.x ver. up")

# first version
## 1. data set
x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

## 2. mdoel set
hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

## 3. session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

## placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

# sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

# second version
## 1. data set
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32)  # shape default, None
# W = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")

## 2. mdoel set
# hypothesis = x_train * W + b
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

## 3. session
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run(
        [cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}
    )
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

## 4. 최종 모델 확인 및 예측
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
