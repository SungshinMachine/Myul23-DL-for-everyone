# Neural이 축삭돌기를 통한 변수들로 함수를 구성하고, 이 함수의 반환값을 thresholding해서 0 또는 1로 활성화되는 형식이라고?
# 초기 인공지능의 관심사는 (2차원에서) AND와 OR를 구분하는 선형식을 발견하고 구성하는 거였는데, XOR는 구분하지 못했다고.
# MLP(Multiple Layer Perceptron)의 weights과 biases의 영향을 제대로 확인할 수 없어 그 값을 지정할 수 없었다.
# 이후 Backpropagarion과 Convolutional 기법이 등장하며 Neural Network가 다시 활성화되고 성장하기 시작했다.
# 역방향 전파 방법의 문제(lots of layers)가 대두되면서 SVM과 RandomForest가 단순함과 그로 인한 높은 설명력으로 각광받았다.
# CIFAR, Canadian Institute for Advanced Research
# (deep-ing) Neural Network -> Deep-learning (2015, about 5 percent images recognition, based on DL)

# 갑자기 저 우주에 점이 나라는 걸 깨달은 천문학 공부자가 된 듯한 기분이네요. 저 커다란 딥러닝의 세계에 전 하나의 점인 듯합니다..
# 물론 아직 딥러닝은 그 구성을 확실하게 이해하지 못해서 아이디어가 떠오르지 않지만, 그럼에도 API가 이용하는 ML은 대충 알 것도 같네요, 다행히.

import tensorflow as tf
import numpy as np

# 김밥이 array... 직업병?
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# pp.pprint(t)  # 쉽게 내용을 보고자 한 것으로 보임.
print(t, f"dimension: {t.ndim}, shape: {t.shape}", sep="\n")
print(t[0], t[1], t[-1])
# 맞다, python의 slicing은 마지막 위치값을 반환하지 않는다. 아마도 range나 i < final로 만들어서겠지.
print(t[2:5], t[4:-1], t[:2], t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
# pp.pprinf(t)
print(t, f"dimension: {t.ndim}, shape: {t.shape}", sep="\n")

# Shape, Rank, Axis
t = tf.constant([1, 2, 3, 4])
tf.shape(t).eval()

t = tf.constant([[1, 2], [3, 4]])
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()  # 1,2,3,4
# Axis, Dimension과 유사한 값을 가지며, 바깥쪽(괄호)부터 axis-i의 값을 갖는다.

# Matmul vs Multiply
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print(f"Matrix 1 shpae: {matrix1.shape}, Matrix 2 shape: {matrix2.shape}",
      sep="\t")
tf.matmul(matrix1, matrix2).eval()
(matrix1 * matrix2).eval()

# Broadcasting, 자동 형변환과 같은 편리함과 단점을 제공
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1 + matrix2).eval()

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
(matrix1 + matrix2).eval()

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
(matrix1 + matrix2).eval()

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
(matrix1 + matrix2).eval()

# Reduce mean 파헤치기
tf.reduce_mean([1, 2], axis=0).eval()
# 원래 데이터가 int였으므로 값을 자동 형변환해 int로 반환

x = [[1., 2.], [3., 4.]]
tf.reduce_mean(x).eval()
tf.reduce_mean(x, axis=0).eval()  # 2, 3
tf.reduce_mean(x, axis=1).eval()  # 1.5, 3.5
tf.reduce_mean(x, axis=-1).eval()  # 1.5, 3.5

# Reduce sum
x = [[1., 2.], [3., 4.]]
tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis=0).eval()  # 4, 6
tf.reduce_sum(x, axis=-1).eval()  # 3, 7
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()

# Argmax
x = [[0, 1, 2], [2, 1, 0]]
tf.argmax(x, axis=0).eval()  # 1,0,0 (index)
tf.argmax(x, axis=1).eval()  # 2, 0
tf.argmax(x, axis=-1).eval()  # 2, 0

# Reshape
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
t.shape
tf.reshape(t, shape=[-1, 3]).eval()
tf.reshape(t, shape=[-1, 1, 3]).eval()

tf.squeeze([[0], [1], [2]]).eval()
# return [0, 1, 2]
tf.expand_dims([0, 1, 2], 1).eval()
# return [[0], [1], [2]]

# One hot
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()

# Casting, 형변환의 그 cast
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()
# 왜 python은 False를 0으로 인식하지 못하는가

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
tf.stack([x, y, z]).eval()  # 행으로 합치기
tf.stack([x, y, x], axis=1).eval()  # transpose 후, 열로 합치기

# Ones and Zeros like
# 그냥 np.ones, np.zeros 쓰면 안 되는 걸까.
x = [[0, 1, 2], [2, 1, 0]]
tf.ones_like(x).eval()
tf.zeros_like(x).eval()

# Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
