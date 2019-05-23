import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
"""
Removed all kinds of regularizers such as dropout and batch_normalization
"""

class Mentee(object):

	def __init__(self, seed):
		self.parameters = []
		self.seed = seed
		self.num_channels = 3
		self.trainable = True

	def fc_student(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			if layerName == "fc1":
				shape = int(np.prod(input.get_shape()[1:]))
			if layerName == "fc2" or layerName == "fc3":
				shape = 4096
			fc_weights = tf.Variable(tf.truncated_normal([shape, out_filter], dtype=tf.float32, stddev=1e-2),trainable=True, name='weights')
			fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=True,name='biases')
			input_flat = tf.reshape(input, [-1, shape])
			fc = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
			relu = tf.nn.relu(fc)
			if layerName == "fc3":
				return fc
			else:
				return relu

	def build_student_oneConvLayer(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			num_filters_in = int(input.shape[3])
			kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters_in, out_filter], dtype=tf.float32, stddev=1e-2, seed=self.seed),trainable=True, name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[out_filter], dtype=tf.float32), trainable=True, name='biases')
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
			out = tf.nn.bias_add(conv, biases)
			out = tf.nn.relu(out, name="relu")
			return out

	def build_student_conv6fc3(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		with tf.name_scope('mentee'):
			conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", 64)
			pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", 128)
			pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", 256)
			pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", 512)
			pool4 = tf.nn.max_pool(conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			conv5_1 = self.build_student_oneConvLayer(pool4, "conv5_1", 512)
			pool5 = tf.nn.max_pool(conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			conv6_1 = self.build_student_oneConvLayer(pool5, "conv6_1", 512)
			pool6 = tf.nn.max_pool(conv6_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')
			print(pool6)

			fc1 = self.fc_student(pool6, "fc1", 4096)
			fc2 = self.fc_student(fc1, "fc2", 4096)
			self.fc3 = self.fc_student(fc2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(fc1)
			print(fc2)
			print(self.fc3)
			return self

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate, global_step):
		# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def _calc_num_trainable_params(self):
		self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
		print('number of trainable params: ' + str(self.num_trainable_params))