import numpy as np
import tensorflow as tf
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _normalize(X):
    means = np.mean(X, axis=0)
    tmp = np.subtract(X, means)
    return tmp, means


def _denormalize(Rn, means):
    return np.add(Rn, means)


def _show_image(lin):
    small_image = np.array(lin).reshape((28, 28))
    plt.imshow(small_image, cmap="gray")
    plt.show()


def _save_image(lin, filename):
    small_image = np.array(lin).reshape((28, 28))
    plt.imsave(filename + '.png', small_image)


if __name__ == '__main__':

    dims = 50
    index = 0

    if len(sys.argv) > 1:
        dims = int(sys.argv[1])
    if len(sys.argv) > 2:
        index = int(sys.argv[2])

    X = np.load('dataset.npy')
    print(X.shape)
    Xn, means = _normalize(X)
    Cov = np.matmul(np.transpose(Xn), Xn)

    Xtf = tf.placeholder(tf.float32, shape=[X.shape[0], X.shape[1]])
    Covtf = tf.placeholder(tf.float32, shape=[Cov.shape[0], Cov.shape[1]])

    Stf, Utf, Vtf = tf.svd(Covtf)
    print(Stf.shape)
    Vtf_T = tf.slice(Vtf, [0, 0], [X.shape[1], dims])

    Ttf = tf.matmul(Xtf, Vtf_T)
    Rtf = tf.matmul(Ttf, Vtf_T, transpose_b=True)

    with tf.Session() as sess:
        Rn = sess.run(Rtf, feed_dict = {
            Xtf: Xn,
            Covtf: Cov
        })

    R = _denormalize(Rn, means)

    # _show_image(X[index])
    # _show_image(R[index])
    _save_image(X[index], 'original')
    _save_image(R[index], '%s_MODES' % dims)
