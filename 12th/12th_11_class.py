# 컴퓨터 성능이 상당히 좋은 게 아니라면, 일반적으론 돌리지 말자.
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 그냥 단일 module보단 package 형태로 만들어보고 싶었다.
from tf_basics import basics as tfb


num_models = 7
training_epochs = 15
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()

models = []
for m in range(num_models):
    models.append(tfb.Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())
print("Learning Started !")

for epoch in range(training_epochs):
    avg_cost = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        for idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost[idx] += c / total_batch

    print("Epoch: %4d" % (epoch + 1), "Cost: ", avg_cost, sep="\n")
print("Learning Finished!")


# Test & 모형 진단
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
# np.zeros도 행렬을 지원했던 것 같은데, 기억이 나지 않습니다.

for idx, m in enumerate(models):
    print(f"{idx} Accucary: {m.get_accuracy(mnist.test.images, mnist.test.labels)}")
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_prediction, tf.float32))
print("Ensemble Accuracy:", sess.run(ensemble_accuracy))
