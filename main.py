import tensorflow as tf
from ResNet101 import resnet_101


def main(argv):
    # TODO provide a (photos) source dir as an argument
    features = tf.placeholder(tf.float16, shape=[None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, shape=[None])
    residual_blocks, logits = resnet_101(features, 1000)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/tmp/resnet", sess.graph)

        writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
