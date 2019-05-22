import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K


class Mentee(object):

    def __init__(self, num_channels, trainable=True):
        self.trainable = trainable
        self.parameters = []
        self.num_channels = num_channels

    def wide_basic(self, nInputPlane, nOutputPlane, seed, train_mode):
        print("wide_basic")
        with tf.name_scope('block') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, nInputPlane, nOutputPlane], dtype=tf.float32,
                                                     stddev=1e-2, seed=seed), trainable=self.trainable,
                                 name='wide_basic_conv1')
            conv = tf.nn.conv2d(nInputPlane, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=self.trainable, name='mentee_biases')
            out = tf.nn.bias_add(conv, biases)
            # out = self.extra_regularization(out)

            self.conv1_1 = tf.nn.relu(out, name=scope)
            # self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
            self.parameters += [kernel, biases]