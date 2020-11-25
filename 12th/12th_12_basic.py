# RNN (Recurrent Neural Network)
# sequence data을 input으로 하며 이전의 처리된 값이 다음 데이터에 영향을 미침을 표현한다.
# 특정 hidden layer를 통해 계산된 값이 다시 같은 layer(연속된 다음 값에 대한 layer)의 변수로 작용한다.
# state: 반복 단계, 시점을 표현해줄 용어이자 해당 단계의 값 (previous/next 개념보단 old/new 사용), 통틀어 cell이라 한다.
# 초기값은 0으로 하시는군요.

# (Vanilla) RNN
# 다음 state로 넘기기: ht = f(h(t - 1), xt);   ht = tanh(W(hh) * h(t - 1) + W(xh) * xt)
# 다음 layer로 넘기기: yt = W(hy) * ht
# others: LSTM, GRU


import numpy as np
import tensorflow as tf


# Loss for Sequence, sequence_loss funciton
# prediction0 = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
# prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
# prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)
# y_data = tf.constant([[1, 1, 1]])
# 문자열의 각 자리에 대한 중요도를 고려할 수 있다.
# weights = tf.constant([[1, 1, 1]], dtype=tf.float32)


# sequence_loss0 = tf.contrib.seq2seq.sequence_loss(logits=prediction0, targets=y_data, weights=weights)
# sequence_loss1 = tf.contrib.seq2seq.sequence_loss(logits=prediction1, targets=y_data, weights=weights)
# sequence_loss2 = tf.contrib.seq2seq.sequence_loss(logits=prediction2, targets=y_data, weights=weights)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(f"Loss0: {sequence_loss0.eval()}", f"Loss1: {sequence_loss1.eval()}", f"Loss2: {sequence_loss2.eval()}", sep="\n")


# 실제
sentence = (
    "if you want to build a ship, don't drum up people together to"
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
)
seq_length = 10  # 한 번 학습에 넣는 문자열의 길이 -1


# unique chars (vocabulary, voc)
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

# python slicing 0부터였던가, 나 왜 R이랑 같다고 기억하고 있지.
# x_data = [sample_idx[:-1]]
# y_data = [sample_idx[1:]]

x_data = []
y_data = []
for i in range(len(sentence) - seq_length):
    # shape을 생각해볼 것, 다음에 올 문자를 예측하는 것이므로 output은 하나 더 가야 한다.
    x_str = sentence[i : i + seq_length]
    y_str = sentence[i + 1 : i + seq_length + 1]

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])


# 모형 구축의 시작
batch_size = len(x_data)  # 학습을 위해 넣은 데이터의 갯수
num_classes = len(char_set)  # 한 문자당 몇 원소로 표현 가능한가
input_dim = num_classes
hidden_size = num_classes

weights = tf.ones([batch_size, seq_length])


X = tf.placeholder(tf.int32, shape=[None, seq_length])
X_one_hot = tf.one_hot(X, num_classes)
Y = tf.placeholder(tf.int32, shape=[None, seq_length])

# 모형 선택 및 구성
# 편의성 등을 위해 hyper-parameter setting을 앞에 빼긴 했다만 원래의 모형 구축을 생각하면 모형 선택 다음에 반복을 통한 hyper-parameter 선정

# cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# cell = tf.contrib.rnn.GRUCell(hidden_size)
# return shape 생각하자.

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
# 원래 RNN으로 나온 데이터를 바로 cost 계산에 넣지 않는다고 합니다.

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # sess.run의 반환값은 list가 기본, 따라서 그 값을 index로 이용하고자 직접적으로 차원을 줄여버림.
        result_str = [char_set[c] for c in np.squeeze(result[0])]
        print(i, "Loss:", l, "Prediction:", "".join(result_str))
