# Matrix를 이용한 vetor implementation를 길게 설명하고 계십니다.
# 총 느낌: 같은 내용 쓸 걸 알았다면 함수화할 걸 그랬다.
import tensorflow as tf

# 1. 원래대로였다면 (비-매트릭스)
## data set
x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

## model set
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

## Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if step % 10 == 0:
        print(step, "Cost :", cost_val, "\nPrediction:", hy_val)


# 2. vector implementation (매트릭스화)
## data set
x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
y_data = [[152], [185], [180], [196], [142]]

# sample의 크기를 확실히 알 수 없으므로, 갯수에서 None를 쓰는 걸 허용해준다.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

## model set
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

## Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost :", cost_val, "\nPrediction:", hy_val)

## Ask
print("Your score will be", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other scores will be", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
# 진짜 code style black으로 한 건데, 코드 길이를 너무 생각해줘서 짜증날 정도..


# addition) load data(file)
# 매번 생각하는 거지만, numpy가 더 구성적으로 단단한가?
# np에 대한 참조가 없는 함수에 대해서 확장이 안돼서 개인적으론 별로라고 생각하는데
# tensorlfow에 float가 np.float32랑 연계되니까 쓰는 거겠지?

# ## data set
# import numpy as np

# xy = np.loadtxt("data-01-test-score.csv", delimiter=",", dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)

# X = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
# Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([x_data.shape[1], 1]))
# b = tf.Variable(tf.random_normal([1]))

# ## model set
# hypothesis = tf.matmul(X, W) + b
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)

# ## Session
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for step in range(2001):
#     cost_val, hy_val, _ = sess.run(
#         [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data}
#     )
#     if step % 10 == 0:
#         print(step, "Cost :", cost_val, "\nPrediction:", hy_val)


# addition) local 용량을 생각해서 read-quere를 이용한 load data(file)
# train.batch도 placeholder처럼 일단 구성하고 구상하고, Session().run()을 통해 돌리게 된다.
# ## Quere 생성
# filename_queue = tf.train.string_input_producer(
#     ["data-01-test-score.csv"],
#     # ["data-01-test-score.csv", "data-02-test-score.csv", ...],
#     shuffle=False,
#     name="filename_queue",
# )

# ## reader 만들기
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)

# ## value parsing
# # 형태와 shape 지정
# record_defaults = [[0.0], [0.0], [0.0], [0.0]]
# xy = tf.decode_csv(value, record_defaults=record_defaults)

# # 위치상 여기가 맞고, tf.train.batch나 tf.train.shuffle_batch 둘 중
# # 하나를 사용할 수 있는 것으로 보이는데 예제를 찾기 어렵다.
# # ## 0. shuffle_batch
# # min_after_dequeue = 10000
# # capacity = min_after_dequeue + 3 * batch_size
# # example_batch, label_batch = tf.train.shuffle_batch(
# #     [example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue
# # )

# ## 1. train용 batch처럼 한 번에 batch_size만큼 불러오기
# train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# ## data set
# X = tf.placeholder(tf.float32, shape=[None, value.shape[1] - 1])
# Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([value.shape[1] - 1, 1]))
# b = tf.Variable(tf.random_normal([1]))

# ## model set
# hypothesis = tf.matmul(X, W) + b
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)

# ## Session
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# for step in range(2001):
#     x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
#     if step % 10 == 0:
#         print(step, "Cost :", cost_val, "\nPrediction:", hy_val)

# coord.request_stop()
# coord.join(threads)
