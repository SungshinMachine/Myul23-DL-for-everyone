# Stacked RNN
# 결과가 좋지 않았다.
import numpy as np
import tensorflow as tf

sentence = (
    "if you want to build a ship, don't drum up people together to"
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
)
seq_length = 10


# unique chars (vocabulary, voc)
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

x_data = []
y_data = []
for i in range(len(sentence) - seq_length):
    x_str = sentence[i : i + seq_length]
    y_str = sentence[i + 1 : i + seq_length + 1]

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])


# 모형 구축의 시작
batch_size = len(x_data)
num_classes = len(char_set)
input_dim = num_classes
hidden_size = num_classes

num_stacked = 5
learning_rate = 0.001

weights = tf.ones([batch_size, seq_length])


# 모형 선택 및 구성
X = tf.placeholder(tf.int32, shape=[None, seq_length])
X_one_hot = tf.one_hot(X, num_classes)
Y = tf.placeholder(tf.int32, shape=[None, seq_length])

# layer
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell] * num_stacked, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
# sequence_length를 list로 줄 수 있다. 난 또 문자열을 split해서 그 길이만큼 seq_length로 해석하는 줄 알았네.
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, sequence_length=seq_length, initial_state=initial_state, dtype=tf.float32)
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

# softmax
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(X_for_softmax, [batch_size, seq_length, num_classes])


sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        results, l, _ = sess.run([outputs, loss, train], feed_dict={X: x_data, Y: y_data})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(f"%3d %3d" % (i, j), "".join([char_set[t] for t in index]), f"\tLoss: {l}")

    results = sess.run(outputs, feed_dict={X: x_data})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:
            print("".join([char_set[t] for t in index]), end="")
        else:
            print(char_set[index[-1]], end="")
