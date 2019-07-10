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

	def fc_student(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			shape = int(np.prod(input.get_shape()[1:]))
			fc_weights = tf.Variable(tf.truncated_normal([shape, out_filter], dtype=tf.float32, stddev=1e-2, seed=self.seed),trainable=True, name='weights')
			fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=True,name='biases')
			input_flat = tf.reshape(input, [-1, shape])
			fc = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
			relu = tf.nn.relu(fc)
			self.parameters += [fc_weights, fc_biases]
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
			self.parameters += [kernel, biases]
			return out

	def build_student_conv5fc1(self, images, num_classes, temp_softmax):
		print("build_student_conv5fc1")
		K.set_learning_phase(True)
		#num_filters = [64, 128, 256, 512, 512] # origin
		num_filters = [64, 128, 256, 512, 512-95] # 100per, 90per
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", num_filters[0])
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", num_filters[1])
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", num_filters[2])
			pool3 = tf.nn.max_pool(self.conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", num_filters[3])
			pool4 = tf.nn.max_pool(self.conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.conv5_1 = self.build_student_oneConvLayer(pool4, "conv5_1", num_filters[4])
			pool5 = tf.nn.max_pool(self.conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			self.fc3 = self.fc_student(pool5, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc3)
			return self

	def build_student_conv6fc3(self, images, num_classes, temp_softmax):
		print("build_student_conv6fc3")
		K.set_learning_phase(True)
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", 64)
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", 128)
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", 256)
			pool3 = tf.nn.max_pool(self.conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", 512)
			pool4 = tf.nn.max_pool(self.conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.conv5_1 = self.build_student_oneConvLayer(pool4, "conv5_1", 512)
			pool5 = tf.nn.max_pool(self.conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			self.conv6_1 = self.build_student_oneConvLayer(pool5, "conv6_1", 512)
			pool6 = tf.nn.max_pool(self.conv6_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')
			print(pool6)

			fc1 = self.fc_student(pool6, "fc1", 4096)
			fc2 = self.fc_student(fc1, "fc2", 4096)
			self.fc3 = self.fc_student(fc2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(fc1)
			print(fc2)
			print(self.fc3)
			return self

	def build_student_conv5fc2(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		num_filters = [64, 128, 256, 512, 512]
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", num_filters[0])
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", num_filters[1])
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", num_filters[2])
			pool3 = tf.nn.max_pool(self.conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", num_filters[3])
			pool4 = tf.nn.max_pool(self.conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.conv5_1 = self.build_student_oneConvLayer(pool4, "conv5_1", num_filters[4])
			pool5 = tf.nn.max_pool(self.conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			fc1 = self.fc_student(pool5, "fc1", 4096)
			self.fc3 = self.fc_student(fc1, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc3)
			return self

	def build_student_conv4fc1(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		num_filters = [64, 128, 256, 512, 512]
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", num_filters[0])
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", num_filters[1])
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", num_filters[2])
			pool3 = tf.nn.max_pool(self.conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", num_filters[3])
			pool4 = tf.nn.max_pool(self.conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.fc3 = self.fc_student(pool4, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc3)
			return self


	def build_student_conv3fc1(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		num_filters = [64, 128, 256, 512, 512]
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", num_filters[0])
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", num_filters[1])
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", num_filters[2])
			pool3 = tf.nn.max_pool(self.conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.fc3 = self.fc_student(pool3, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc3)
			return self

	def build_student_conv2fc1(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		num_filters = [64, 128, 256, 512, 512]
		with tf.name_scope('mentee'):
			self.conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", num_filters[0])
			pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", num_filters[1])
			pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.fc3 = self.fc_student(pool2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc3)
			return self

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate, global_step):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def _calc_num_trainable_params(self):
		self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
		print('number of trainable params: ' + str(self.num_trainable_params))